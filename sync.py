# Importing necessary libraries
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import os

# =====================
# 1. Data Loading
# =====================
def load_data(filename='results.csv'):
    """
    Load data from a CSV file in the same directory as the script.
    Args:
        filename (str): Name of the CSV file containing the data

    Returns:
        numpy.ndarray: Array of packet per second values
    """
    # Getting the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Constructing the full file path
    file_path = os.path.join(script_dir, filename)

    # Loading the data
    df = pd.read_csv(file_path)
    print("Data loaded successfully. First few rows:")
    print(df.head())

    # Returning the packet per second values as a numpy array
    return df["packets_per_sec"].values

# Loading the data
data = load_data()
dt = 1  # Time steps in seconds

# =====================
# 2. Synchronization Methods
# =====================
def adaptive_event_sync(pt, initial_thr, drift, adaptive_factor=0.1):
    """
    Adaptive event-based synchronization method.

    Args:
        pt (numpy.ndarray): Input time series data
        initial_thr (float): Initial threshold for synchronization
        drift (float): Drift rate when not synchronizing
        adaptive_factor (float): Factor for adapting the threshold

    Returns:
        tuple: (synchronized_state, synchronization_indices)
    """
    # Initializing state and synchronization indices
    state, idx = [pt[0]], []
    thr = initial_thr  

    # Iterating through the time series
    for i in range(1, len(pt)):
        # Checking if current value differs from last state by more than threshold
        if abs(pt[i] - state[-1]) > thr:
            # Synchronizing to current value
            state.append(pt[i])
            idx.append(i)
            thr = max(initial_thr, thr - adaptive_factor)
        else:
            # Drifting the state if not synchronizing
            state.append(state[-1] + drift * dt)
            thr = min(initial_thr * 2, thr + adaptive_factor)

    return np.array(state), idx

def mpc_sync(pt, drift, horizon, cost, max_drift_error=1.0):
    """
    Model Predictive Control synchronization method with a hard cap on drift error.

    Args:
        pt (np.ndarray): Input time series data
        drift (float): Drift rate when not synchronizing
        horizon (int): Prediction horizon
        cost (float): Cost of synchronization
        max_drift_error (float): If abs(error) exceeds this, force a sync

    Returns:
        tuple: (synchronized_state, synchronization_indices)
    """
    state, idx = [pt[0]], []

    for i in range(1, len(pt)):
        pred_state = state[-1] + drift * dt
        err = abs(pt[i] - pred_state)

        # if drift-error would exceed max_drift_error, we force a sync
        if err > max_drift_error:
            state.append(pt[i])
            idx.append(i)
            continue

        # otherwise use the original MPC cost check
        c_sync = cost + abs(pt[i] - state[-1])
        c_drift = sum(
            abs(pt[i + h] - (state[-1] + drift * (h + 1) * dt))
            for h in range(horizon) if i + h < len(pt)
        )

        if c_sync < c_drift:
            state.append(pt[i])
            idx.append(i)
        else:
            state.append(pred_state)

    return np.array(state), idx


def kalman_sync(pt, proc_var, meas_var):
    """
    Kalman Filter synchronization method.

    Args:
        pt (numpy.ndarray): Input time series data
        proc_var (float): Process noise variance
        meas_var (float): Measurement noise variance

    Returns:
        tuple: (synchronized_state, synchronization_indices)
    """
    # Initializing Kalman filter
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([pt[0]])
    kf.F = kf.H = np.array([[1]]) 
    kf.P *= 1000
    kf.Q = proc_var
    kf.R = meas_var

    # Initializing state and synchronization indices
    state, idx = [pt[0]], []

    # Iterating through the time series
    for i, z in enumerate(pt[1:], 1):
        kf.predict()

        # Checking if measurement is significantly different from prediction
        if abs(z - kf.x[0]) > 3 * np.sqrt(kf.P[0, 0]):
            kf.x = np.array([z])
            idx.append(i)

        kf.update(z)
        state.append(kf.x[0])

    return np.array(state), idx

# =====================
# 3. RL Environment & Training
# =====================
class SyncEnv(gym.Env):
    """
    Custom Gym environment for synchronization tasks.
    """
    def __init__(self, pt, window=5):
        super().__init__()
        self.pt = pt  # Input time series
        self.window = window  # Window size for calculating variance

        # Defining action and observation spaces
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)

        # Precomputing derivative and variance
        self._deriv = np.gradient(pt)
        self._var = np.array([np.var(pt[max(0,i-window):i+1]) for i in range(len(pt))])

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        """
        super().reset(seed=seed)
        # Resetting the environment
        self.t = 0  # Current time step
        self.state = self.pt[0]
        self.last = 0  # Last synchronization time
        self.last_action = 0.0  # Last action taken
        return self._obs(), {}

    def _obs(self):
        """
        Getting current observation.
        """
        # Getting the current observation
        err = self.pt[self.t] - self.state
        time_since = (self.t - self.last) * dt
        return np.array([err, time_since, self.last_action, self._deriv[self.t], self._var[self.t]], np.float32)

    def step(self, action):
        """
        Execute one time step in the environment.
        """
        # Converting action to threshold factor
        factor = float(np.clip(action, 0, 1))
        thr = 1.0 + factor * 4.0

        # Calculating current error
        err = abs(self.pt[self.t] - self.state)

        # Determining if we should synchronize
        sync = err > thr

        if sync:
            # Synchronizing
            self.state = self.pt[self.t]
            self.last = self.t
            event_bonus = 0.5
        else:
            # Drifting the state
            self.state += 0.05 * dt
            event_bonus = 0.0

        # Calculating reward
        reward = -err**2 + event_bonus - 0.2*factor

        # Updating last action
        self.last_action = factor

        # Checking if episode is done
        done = self.t >= len(self.pt)-2

        # Moving to next time step
        self.t += 1

        return self._obs(), reward, done, False, {}

def train_rl(pt, algo='PPO', timesteps=300_000):
    """
    Train an RL model for synchronization.

    Args:
        pt (numpy.ndarray): Input time series data
        algo (str): RL algorithm to use ('PPO' or 'SAC')
        timesteps (int): Number of training timesteps

    Returns:
        tuple: (trained_model, environment)
    """
    # Creating environment function
    def make_env():
        return Monitor(SyncEnv(pt))

    # Creating vectorized environment
    venv = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False)

    # Creating RL model
    if algo == 'PPO':
        model = PPO('MlpPolicy', venv, learning_rate=5e-5, n_steps=2048, batch_size=128, n_epochs=10,
                    gamma=0.995, gae_lambda=0.9, policy_kwargs=dict(net_arch=[256,256]), verbose=1)
    else:
        model = SAC('MlpPolicy', venv, learning_rate=3e-4, buffer_size=100_000, batch_size=256,
                    tau=0.005, gamma=0.99, policy_kwargs=dict(net_arch=[256,256]), verbose=1)

    # Creating evaluation callback
    cb = EvalCallback(venv, best_model_save_path='./best', eval_freq=10_000,
                      callback_after_eval=StopTrainingOnNoModelImprovement(20,30))

    # Training the model
    model.learn(total_timesteps=timesteps, callback=cb)

    return model, venv

# =====================
# 4. RL-based Sync
# =====================
def rl_event_sync(pt, drift, rl_model, venv):
    """
    RL-based synchronization method.

    Args:
        pt (numpy.ndarray): Input time series data
        drift (float): Drift rate when not synchronizing
        rl_model: Trained RL model
        venv: Vectorized environment

    Returns:
        tuple: (synchronized_state, synchronization_indices)
    """
    # Initializing state and synchronization indices
    state, idx = [pt[0]], []  # Initializing state and synchronization indices
    obs = venv.reset()  # Resetting environment

    # Iterating through the time series
    for i in range(1, len(pt)):
        # Getting action from RL model
        action, _ = rl_model.predict(obs, deterministic=True)

        # Stepping the environment
        obs, _, _, _ = venv.step(action)

        # Converting action to threshold factor
        factor = float(np.clip(action, 0, 1))
        thr = 1 + factor * 4

        # Checking if we should synchronize
        if abs(pt[i] - state[-1]) > thr:
            # Synchronizing
            state.append(pt[i])
            idx.append(i)
        else:
            # Drifting
            state.append(state[-1] + drift * dt)  # Drift

    return np.array(state), idx

# =====================
# 5. Evaluation and visualization
# =====================
def evaluate(pt, methods):
    """
    Evaluate synchronization methods.

    Args:
        pt (numpy.ndarray): Input time series data
        methods (dict): Dictionary of synchronization methods

    Returns:
        list: List of evaluation results for each method
    """
    results = []

    # Evaluating each method
    for name, fn in methods.items():
        state, idx = fn()

        # Calculating metrics
        mae = np.mean(np.abs(pt - state))
        rmse = np.sqrt(np.mean((pt - state)**2))

        results.append({
            'method': name,
            'mae': mae,
            'rmse': rmse,
            'syncs': len(idx)
        })

    return results

def plot_comparison(results, save_dir, dpi=600):
    names    = [r['method'] for r in results]
    mae_vals = [r['mae']    for r in results]
    rmse_vals= [r['rmse']   for r in results]

    bar_width = 0.35
    r1 = np.arange(len(names))
    r2 = r1 + bar_width

    fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)

    # Ploting with Stronger colors, bold edges
    bars1 = ax.bar(r1, mae_vals, width=bar_width, label='MAE', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(r2, rmse_vals, width=bar_width, label='RMSE', edgecolor='black', linewidth=1.2)

    # Labels & Bold formatting
    ax.set_xlabel('Synchronization method', fontsize=18, fontweight='bold')
    ax.set_ylabel('Error magnitude',       fontsize=18, fontweight='bold')
    ax.set_title('Performance comparison: MAE vs RMSE', fontsize=22, fontweight='bold')

    ax.set_xticks(r1 + bar_width/2)
    ax.set_xticklabels(names, fontsize=16, fontweight='bold')

    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

    # Value labels
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01*h,
                f'{h:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    legend = ax.legend(fontsize=16, frameon=True)
    legend.get_title() and legend.set_title(None)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison_hd.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_sync_counts(results, save_dir, dpi=600):
    names  = [r['method'] for r in results]
    counts = [r['syncs']  for r in results]

    fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)

    bars = ax.bar(names, counts, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Synchronization method', fontsize=18, fontweight='bold')
    ax.set_ylabel('Number of Sync events', fontsize=18, fontweight='bold')
    ax.set_title('Synchronization event frequency', fontsize=22, fontweight='bold')

    ax.set_xticklabels(names, fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01*h,
                f'{h:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sync_counts_hd.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_cumulative_error(pt, methods, save_dir, dpi=600):
    """Enhanced cumulative error plot with high-definition dashed lines"""
    t = np.arange(len(pt))

    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)

    
    colors = {
        'Adaptive Event': '#2E86AB',
        'MPC': '#A23B72', 
        'Kalman': '#F18F01',
        'RL': '#C73E1D'
    }
    
    
    line_styles = {
        'Adaptive Event': (0, (12, 6)),      
        'MPC': (0, (16, 8, 4, 8)),          
        'Kalman': (0, (8, 4)),              
        'RL': (0, (20, 6, 4, 6, 4, 6))     
    }

    for name, fn in methods.items():
        state, _ = fn()
        cum_err = np.cumsum(np.abs(pt - state))
        
        ax.plot(t, cum_err, label=name, 
                color=colors.get(name, '#333333'),
                linestyle=line_styles.get(name, '--'),
                linewidth=6.5, alpha=0.9)

    ax.set_title('Cumulative absolute error over time', fontsize=28, fontweight='bold', pad=30)
    ax.set_xlabel('Time (seconds)', fontsize=24, fontweight='bold', labelpad=20)
    ax.set_ylabel('Cumulative error', fontsize=24, fontweight='bold', labelpad=20)

    ax.tick_params(axis='both', labelsize=18, width=2, length=10)
    ax.tick_params(axis='both', which='minor', width=1.5, length=6)
    
    # legend
    legend = ax.legend(fontsize=20, frameon=True, fancybox=True, shadow=True,
                      loc='upper left', framealpha=0.95)
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')

    ax.grid(True, alpha=0.4, linewidth=1.5)
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cumulative_error_hd.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


# =====================
# 6. Information transfer analysis
# =====================
def calculate_information_metrics(pt, methods, bits_per_sync=32):
    """
    Calculate information transfer metrics for each synchronization method.

    Args:
        pt (numpy.ndarray): Input time series data
        methods (dict): Dictionary of synchronization methods
        bits_per_sync (int): Number of bits needed for each synchronization event

    Returns:
        list: List of information metrics for each method
    """
    info_metrics = []

    # Calculating information metrics for each method
    for name, fn in methods.items():
        state, idx = fn()

        # Calculating metrics
        total_bits = len(idx) * bits_per_sync
        bits_per_time = total_bits / (len(pt) * dt)
        mae = np.mean(np.abs(pt - state))
        error_info_tradeoff = mae * (total_bits / len(pt))

        info_metrics.append({
            'method': name,
            'total_bits': total_bits,
            'bits_per_time': bits_per_time,
            'error_info_tradeoff': error_info_tradeoff,
            'mae': mae
        })

    return info_metrics

def plot_information_metrics(info_metrics, save_dir, dpi=600):
    """
    Plot different metrics of information transmission for various synchronization methods.

    Args:
        info_metrics (list of dict): A list of dictionaries, each containing metrics such as:
            - 'method': Name of the synchronization method
            - 'total_bits': Total number of bits transmitted
            - 'bits_per_time': Bits transmitted per unit time
            - 'error_info_tradeoff': Product of MAE and bits/sample for trade-off analysis
        save_dir (str): Path to the directory where the plot will be saved
        dpi (int, optional): Resolution of the saved image. Defaults to 600.

    Description:
        This function generates a 3-panel bar chart showing:
            1. Total bits transmitted by each method
            2. Communication bandwidth (bits per time unit)
            3. Trade-off between error and information transfer (MAE × bits/sample)

        Each bar is annotated with its value, and plots are styled for clarity and publication quality.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Extracting data
    methods = [m['method'] for m in info_metrics]
    total_bits = [m['total_bits'] for m in info_metrics]
    bits_per_time = [m['bits_per_time'] for m in info_metrics]
    error_info = [m['error_info_tradeoff'] for m in info_metrics]

    # Creating figure with enhanced sizing
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), dpi=dpi)

    # Enhanced color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # 1. Total bits transmitted
    bars1 = axes[0].bar(methods, total_bits, color=colors[:len(methods)], alpha=0.85,
                       edgecolor='black', linewidth=2.5)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')
    
    axes[0].set_title('Total Information transmitted (bits)',
                     fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    axes[0].set_ylabel('Bits', fontsize=16, fontweight='bold', color='#2c3e50', labelpad=15)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[0].set_ylim(0, max(total_bits) * 1.2)

    # 2. Bits per time unit
    bars2 = axes[1].bar(methods, bits_per_time, color=colors[:len(methods)], alpha=0.85,
                       edgecolor='black', linewidth=2.5)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                    f'{height:.1f}', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')
    
    axes[1].set_title('Communication bandwidth (bits per time unit)',
                     fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    axes[1].set_ylabel('Bits/Time', fontsize=16, fontweight='bold', color='#2c3e50', labelpad=15)
    axes[1].set_ylim(0, max(bits_per_time) * 1.2)

    # 3. Error-Information tradeoff
    bars3 = axes[2].bar(methods, error_info, color=colors[:len(methods)], alpha=0.85,
                       edgecolor='black', linewidth=2.5)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')
    
    axes[2].set_title('Error-Information tradeoff',
                     fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    axes[2].set_ylabel('MAE × Bits/sample', fontsize=16, fontweight='bold', color='#2c3e50', labelpad=15)
    axes[2].set_ylim(0, max(error_info) * 1.2)

    # Enhanced styling for all subplots
    for ax in axes:
        ax.set_xlabel('Synchronization method', fontsize=16, fontweight='bold', color='#2c3e50', labelpad=15)
        ax.tick_params(axis='x', labelsize=14, rotation=15, width=2, length=8)
        ax.tick_params(axis='y', labelsize=14, width=2, length=8)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.2)
        ax.set_axisbelow(True)
        
        # Enhanced spines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.suptitle('Information transfer analysis',
                fontsize=22, fontweight='bold', color='#2c3e50', y=0.98)
    plt.subplots_adjust(top=0.88)

    # Saving the plot
    plot_path = os.path.join(save_dir, 'information_metrics.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_error_vs_information(info_metrics, save_dir, dpi=600):
    """
    Plotting the trade-off between prediction error and information transfer for different methods.

    Args:
        info_metrics (list of dict): Each dict should contain 'method', 'mae', and 'total_bits'.
        save_dir (str): Directory where the plot will be saved.
        dpi (int): Resolution of the saved plot.

    Returns:
        None. Saves the plot as 'error_vs_information.png' in the specified directory.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)

    # Extracting data
    methods = [m['method'] for m in info_metrics]
    mae = [m['mae'] for m in info_metrics]
    bits_per_sample = [m['total_bits']/len(data) for m in info_metrics]

    # Colors for scatter points
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Creating enhanced scatter plot
    for i, (method, x, y) in enumerate(zip(methods, bits_per_sample, mae)):
        ax.scatter(x, y, s=250, color=colors[i % len(colors)], alpha=0.8,
                  edgecolor='black', linewidth=2.5, label=method, zorder=5)
        ax.text(x+0.025, y, method, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Plot styling
    ax.set_title('Error vs Information transfer trade-off',
                fontsize=22, fontweight='bold', color='#2c3e50', pad=25)
    ax.set_xlabel('Information transfer (bits per sample)',
                fontsize=18, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Mean Absolute Error (MAE)',
                fontsize=18, fontweight='bold', color='#2c3e50', labelpad=15)
    
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.2)
    ax.set_axisbelow(True)

    # Enhanced explanatory text
    ax.text(0.02, 0.02,
            "Better methods are closer to the origin (lower error, lower bits)",
            transform=ax.transAxes, fontsize=14, fontstyle='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    # Setting axis limits with padding
    x_max = max(bits_per_sample) * 1.25
    y_max = max(mae) * 1.25
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_facecolor('#fafafa')
    
    ax.tick_params(axis='both', labelsize=14, width=2, length=8)

    plt.tight_layout()

    # Saving the plot
    plot_path = os.path.join(save_dir, 'error_vs_information.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# =====================
# 7. Information Transmission tracking
# =====================
def track_information_transmission(pt, methods, bits_per_sync=32):
    """
    Track information transmission at each synchronization event.

    Args:
        pt (numpy.ndarray): Input time series data
        methods (dict): Dictionary of synchronization methods
        bits_per_sync (int): Number of bits needed for each synchronization event

    Returns:
        dict: Dictionary with information transmission data for each method
    """
    info_tracking = {}

    # Tracking information transmission for each method
    for name, fn in methods.items():
        state, idx = fn()

        # Tracking information transmission
        sync_times = [i * dt for i in idx]
        cumulative_bits = [bits_per_sync * (i+1) for i in range(len(idx))]
        sync_values = [pt[i] for i in idx]

        info_tracking[name] = {
            'sync_times': sync_times,
            'cumulative_bits': cumulative_bits,
            'sync_values': sync_values,
            'total_bits': bits_per_sync * len(idx)
        }

    return info_tracking

def plot_information_per_sync_event_hd(info_tracking, save_dir, dpi=600):
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # Creating figure with enhanced higher DPI
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)

    # Enhanced color palette
    colors = {
        'Adaptive Event': '#2E86AB',
        'MPC': '#A23B72',
        'Kalman': '#F18F01',
        'RL': '#C73E1D'
    }

    # Dash patterns - bigger and more visible
    line_styles = {
        'Adaptive Event': (0, (15, 8)),          
        'MPC': (0, (18, 10, 4, 10)),            
        'Kalman': (0, (10, 5)),                 
        'RL': (0, (25, 8, 5, 8, 5, 8))         
    }

    # Plotting each method with enhanced styling (no markers)
    for name, data in info_tracking.items():
        ax.plot(data['sync_times'], data['cumulative_bits'],
               label=name, color=colors.get(name, '#333333'),
               linestyle=line_styles.get(name, '--'),
               linewidth=6.5, alpha=0.9, zorder=2)  

    # Title and labels
    ax.set_title('Information transmitted at each synchronization event',
                fontsize=28, fontweight='bold', color='#2c3e50', pad=35)
    ax.set_xlabel('Time (seconds)', fontsize=24, fontweight='bold', 
                 color='#2c3e50', labelpad=20)
    ax.set_ylabel('Cumulative bits transmitted', fontsize=24, fontweight='bold', 
                 color='#2c3e50', labelpad=20)

    # grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.5)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(loc='upper left', fontsize=18, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.95,
                      edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(2)

    # Setting axis limits with enhanced padding
    if info_tracking:
        max_time = max(max(data['sync_times']) for data in info_tracking.values() if data['sync_times'])
        max_bits = max(max(data['cumulative_bits']) for data in info_tracking.values() if data['cumulative_bits'])
        ax.set_xlim(0, max_time * 1.08)
        ax.set_ylim(0, max_bits * 1.12)

    # Enhanced styling
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=18, width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', width=1.5, length=6)

    # Face color
    ax.set_facecolor('#f8f9fa')

    # Using tight layout
    plt.tight_layout()

    # Saving the enhanced plot
    plot_path = os.path.join(save_dir, 'information_transmission_hd.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

# =====================
# 8. Main execution
# =====================
if __name__ == '__main__':
    # Getting the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Training RL model
    print("Training RL model...")
    ppo_model, ppo_env = train_rl(data, algo='PPO', timesteps=300_000)

    # Defining synchronization methods to evaluate
    methods = {
        'Adaptive Event': lambda: adaptive_event_sync(data, 2, 0.05),
        'MPC': lambda: mpc_sync(data, 0.05, 5, 3),
        'Kalman': lambda: kalman_sync(data, 0.1, 1),
        'RL': lambda: rl_event_sync(data, 0.05, ppo_model, ppo_env)
    }

    # Tracking information transmission
    print("Tracking information transmission...")
    info_tracking = track_information_transmission(data, methods)

    # Plotting information per sync event
    print("Plotting information per sync event...")
    plot_information_per_sync_event_hd(info_tracking, script_dir)

    # Evaluating methods
    print("Evaluating methods...")
    results = evaluate(data, methods)

    # Printing results
    print("\n=== Global Metrics ===")
    for r in results:
        print(f"{r['method']}: MAE={r['mae']:.3f}, RMSE={r['rmse']:.3f}, Syncs={r['syncs']}")

    # Plotting results
    print("\nPlotting results...")
    plot_comparison(results, script_dir)
    plot_sync_counts(results, script_dir)
    plot_cumulative_error(data, methods, script_dir)

    # Calculating and plotting information metrics
    print("\nCalculating information metrics...")
    info_metrics = calculate_information_metrics(data, methods)
    plot_information_metrics(info_metrics, script_dir)
    plot_error_vs_information(info_metrics, script_dir)

    print("\nDone! All plots have been saved to the script directory.")
