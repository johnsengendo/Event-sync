# Import necessary libraries
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
dt = 1  # Time step

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
    state, idx = [pt[0]], []  # Initialize state and synchronization indices
    thr = initial_thr  # Initialize threshold

    # Iterating through the time series
    for i in range(1, len(pt)):
        # Checking if current value differs from last state by more than threshold
        if abs(pt[i] - state[-1]) > thr:
            # Synchronizing to current value
            state.append(pt[i])  # Synchronize to current value
            idx.append(i)  # Record synchronization index
            thr = max(initial_thr, thr - adaptive_factor)  # Decrease threshold
        else:
            # Drifting the state if not synchronizing
            state.append(state[-1] + drift * dt)
            thr = min(initial_thr * 2, thr + adaptive_factor)  # Increase threshold

    return np.array(state), idx

def mpc_sync(pt, drift, horizon, cost):
    """
    Model Predictive Control synchronization method.

    Args:
        pt (numpy.ndarray): Input time series data
        drift (float): Drift rate when not synchronizing
        horizon (int): Prediction horizon
        cost (float): Cost of synchronization

    Returns:
        tuple: (synchronized_state, synchronization_indices)
    """
    # Initializing state and synchronization indices
    state, idx = [pt[0]], []  # Initialize state and synchronization indices

    # Iterating through the time series
    for i in range(1, len(pt)):
        # Calculating cost of synchronizing now
        c_sync = cost + abs(pt[i] - state[-1])

        # Calculating cost of drifting for the prediction horizon
        c_drift = sum(
            abs(pt[i + h] - (state[-1] + drift * (h + 1) * dt))
            for h in range(horizon) if i + h < len(pt)
        )

        # Choosing the action with lower cost
        if c_sync < c_drift:
            # Synchronizing
            state.append(pt[i])  # Synchronize
            idx.append(i)  # Record synchronization index
        else:
            # Drifting
            state.append(state[-1] + drift * dt)  # Drift

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
    # Initializing Kalman Filter
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([pt[0]])  # Initial state
    kf.F = kf.H = np.array([[1]])  # State transition and measurement matrices
    kf.P *= 1000  # Initial state covariance
    kf.Q = proc_var  # Process noise covariance
    kf.R = meas_var  # Measurement noise covariance

    # Initializing state and synchronization indices
    state, idx = [pt[0]], []  # Initialize state and synchronization indices

    # Iterating through the time series
    for i, z in enumerate(pt[1:], 1):
        kf.predict()  # Predict next state

        # Checking if measurement is significantly different from prediction
        if abs(z - kf.x[0]) > 3 * np.sqrt(kf.P[0, 0]):
            kf.x = np.array([z])  # Reset state to measurement
            idx.append(i)  # Record synchronization index

        kf.update(z)  # Update state with measurement
        state.append(kf.x[0])  # Record state

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
        self.state = self.pt[0]  # Current state
        self.last = 0  # Last synchronization time
        self.last_action = 0.0  # Last action taken
        return self._obs(), {}

    def _obs(self):
        """
        Get current observation.
        """
        # Getting the current observation
        err = self.pt[self.t] - self.state  # Current error
        time_since = (self.t - self.last) * dt  # Time since last sync
        return np.array([err, time_since, self.last_action, self._deriv[self.t], self._var[self.t]], np.float32)

    def step(self, action):
        """
        Execute one time step in the environment.
        """
        # Converting action to threshold factor
        factor = float(np.clip(action, 0, 1))
        thr = 1.0 + factor * 4.0  # Calculate threshold

        # Calculating current error
        err = abs(self.pt[self.t] - self.state)

        # Determining if we should synchronize
        sync = err > thr

        if sync:
            # Synchronizing
            self.state = self.pt[self.t]  # Synchronize to current value
            self.last = self.t  # Update last sync time
            event_bonus = 0.5  # Bonus for synchronizing
        else:
            # Drifting the state
            self.state += 0.05 * dt  # Drift the state
            event_bonus = 0.0  # No bonus for drifting

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
    state, idx = [pt[0]], []  # Initialize state and synchronization indices
    obs = venv.reset()  # Reset environment

    # Iterating through the time series
    for i in range(1, len(pt)):
        # Getting action from RL model
        action, _ = rl_model.predict(obs, deterministic=True)

        # Stepping the environment
        obs, _, _, _ = venv.step(action)

        # Converting action to threshold factor
        factor = float(np.clip(action, 0, 1))
        thr = 1 + factor * 4  # Calculate threshold

        # Checking if we should synchronize
        if abs(pt[i] - state[-1]) > thr:
            # Synchronizing
            state.append(pt[i])  # Synchronize
            idx.append(i)  # Record synchronization index
        else:
            # Drifting
            state.append(state[-1] + drift * dt)  # Drift

    return np.array(state), idx

# =====================
# 5. Evaluation and Visualization
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
        state, idx = fn()  # Get synchronized state and indices

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

def plot_comparison(results, save_dir):
    """
    Plot comparison of MAE and RMSE for different methods.

    Args:
        results (list): Evaluation results
        save_dir (str): Directory to save the plot
    """
    # Preparing data for plotting
    names = [r['method'] for r in results]
    mae_vals = [r['mae'] for r in results]
    rmse_vals = [r['rmse'] for r in results]

    plt.style.use('seaborn-v0_8-whitegrid')

    bar_width = 0.35
    r1 = np.arange(len(names))
    r2 = r1 + bar_width

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plotting MAE bars
    bars1 = ax.bar(r1, mae_vals, width=bar_width, label='MAE',
                   color='#00008B', alpha=0.8, edgecolor='black', linewidth=0.7)
    # Plotting RMSE bars
    bars2 = ax.bar(r2, rmse_vals, width=bar_width, label='RMSE',
                   color='#FFA500', alpha=0.8, edgecolor='black', linewidth=0.7)

    # Adding value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Setting labels and title
    ax.set_xlabel('Synchronization Method', fontsize=14, fontweight='bold',
                  color='#2c3e50', labelpad=10)
    ax.set_ylabel('Error Magnitude', fontsize=14, fontweight='bold',
                  color='#2c3e50', labelpad=10)
    ax.set_title('Performance Comparison: MAE and RMSE',
                 fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

    ax.set_xticks(r1 + bar_width/2)
    ax.set_xticklabels(names, fontsize=12, fontweight='bold')

    ax.tick_params(axis='y', labelsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

    legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    y_max = max(max(mae_vals), max(rmse_vals))
    ax.set_ylim(0, y_max * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Saving the plot
    plot_path = os.path.join(save_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sync_counts(results, save_dir):
    """
    Plot synchronization event counts for different methods.

    Args:
        results (list): Evaluation results
        save_dir (str): Directory to save the plot
    """
    # Preparing data for plotting
    names = [r['method'] for r in results]
    sync_counts = [r['syncs'] for r in results]

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plotting sync counts
    colors = ['#00008B', '#00008B', '#00008B', '#00008B']

    bars = ax.bar(names, sync_counts, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.7, width=0.6)

    for bar, count in zip(bars, sync_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='#2c3e50')

    # Setting labels and title
    ax.set_xlabel('Synchronization Method', fontsize=14, fontweight='bold',
                  color='#2c3e50', labelpad=10)
    ax.set_ylabel('Number of Synchronization Events', fontsize=14, fontweight='bold',
                  color='#2c3e50', labelpad=10)
    ax.set_title('Synchronization event frequency by method',
                 fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    y_max = max(sync_counts)
    ax.set_ylim(0, y_max * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    ax.set_facecolor('#fafafa')

    # Saving the plot
    plot_path = os.path.join(save_dir, 'sync_counts.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_error(pt, methods, save_dir):
    """
    Plot cumulative error over time for different methods.

    Args:
        pt (numpy.ndarray): Input time series data
        methods (dict): Dictionary of synchronization methods
        save_dir (str): Directory to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    t = np.arange(0, len(pt)) * dt

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    linestyles = ['-', '--', '-.', ':']
    linewidths = [2.5, 2.5, 2.5, 2.5]

    # Plotting cumulative error for each method
    for i, (name, fn) in enumerate(methods.items()):
        state, _ = fn()
        error = np.abs(pt - state)
        cum_error = np.cumsum(error)
        ax.plot(t, cum_error, color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=linewidths[i % len(linewidths)],
                label=f'{name}', alpha=0.9)

    # Setting title and labels
    ax.set_title('Cumulative Absolute Error Over Time',
                 fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold',
                  color='#2c3e50', labelpad=10)
    ax.set_ylabel('Cumulative absolute error', fontsize=14, fontweight='bold',
                  color='#2c3e50', labelpad=10)

    ax.tick_params(axis='both', labelsize=11)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    legend = ax.legend(loc='upper left', fontsize=12, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    ax.set_facecolor('#fafafa')

    # Saving the plot
    plot_path = os.path.join(save_dir, 'cumulative_error.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# =====================
# 6. Information Transfer Analysis
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

def plot_information_metrics(info_metrics, save_dir):
    """
    Plot information transfer metrics.

    Args:
        info_metrics (list): Information metrics for each method
        save_dir (str): Directory to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Extracting data
    methods = [m['method'] for m in info_metrics]
    total_bits = [m['total_bits'] for m in info_metrics]
    bits_per_time = [m['bits_per_time'] for m in info_metrics]
    error_info = [m['error_info_tradeoff'] for m in info_metrics]

    # Creating figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    # 1. Total bits transmitted
    bars1 = axes[0].bar(methods, total_bits, color='#4C72B0', alpha=0.8,
                       edgecolor='black', linewidth=0.7)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    axes[0].set_title('Total Information Transmitted (bits)',
                     fontsize=14, fontweight='bold', color='#2c3e50')
    axes[0].set_ylabel('Bits', fontsize=12, fontweight='bold', color='#2c3e50')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[0].set_ylim(0, max(total_bits) * 1.15)

    # 2. Communication Bandwidth
    bars2 = axes[1].bar(methods, bits_per_time, color='#C44E52', alpha=0.8,
                       edgecolor='black', linewidth=0.7)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                    f'{height:.1f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    axes[1].set_title('Communication Bandwidth (bits per time unit)',
                     fontsize=14, fontweight='bold', color='#2c3e50')
    axes[1].set_ylabel('Bits/Time', fontsize=12, fontweight='bold', color='#2c3e50')
    axes[1].set_ylim(0, max(bits_per_time) * 1.15)

    # 3. Error-Information Tradeoff
    bars3 = axes[2].bar(methods, error_info, color='#8172B3', alpha=0.8,
                       edgecolor='black', linewidth=0.7)
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    axes[2].set_title('Error-Information Tradeoff (lower is better)',
                     fontsize=14, fontweight='bold', color='#2c3e50')
    axes[2].set_ylabel('MAE × Bits/Sample', fontsize=12, fontweight='bold', color='#2c3e50')
    axes[2].set_ylim(0, max(error_info) * 1.15)

    # Applying common styling for all subplots
    for ax in axes:
        ax.set_xlabel('Synchronization Method', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.suptitle('Information Transfer Analysis',
                fontsize=16, fontweight='bold', color='#2c3e50', y=1.02)
    plt.subplots_adjust(top=0.85)

    # Saving the plot
    plot_path = os.path.join(save_dir, 'information_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_vs_information(info_metrics, save_dir):
    """
    Plot error vs information tradeoff.

    Args:
        info_metrics (list): Information metrics for each method
        save_dir (str): Directory to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Extracting data
    methods = [m['method'] for m in info_metrics]
    mae = [m['mae'] for m in info_metrics]
    bits_per_sample = [m['total_bits']/len(data) for m in info_metrics]

    # Defining colors for scatter points
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # Creating scatter plot
    for i, (method, x, y) in enumerate(zip(methods, bits_per_sample, mae)):
        ax.scatter(x, y, s=180, color=colors[i % len(colors)], alpha=0.8,
                  edgecolor='black', linewidth=1, label=method)
        ax.text(x+0.02, y, method, fontsize=12, fontweight='bold')

    # Enhancing plot styling
    ax.set_title('Error vs Information Transfer Trade-off',
                fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax.set_xlabel('Information Transfer (bits per sample)',
                fontsize=14, fontweight='bold', color='#2c3e50', labelpad=10)
    ax.set_ylabel('Mean Absolute Error (MAE)',
                fontsize=14, fontweight='bold', color='#2c3e50', labelpad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adding explanatory text
    ax.text(0.02, 0.02,
            "Better methods are closer to the origin (lower error, lower bits)",
            transform=ax.transAxes, fontsize=12, fontstyle='italic')

    # Setting axis limits with some padding
    x_max = max(bits_per_sample) * 1.2
    y_max = max(mae) * 1.2
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Removing top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Adding subtle background color
    ax.set_facecolor('#fafafa')

    plt.tight_layout()

    # Saving the plot
    plot_path = os.path.join(save_dir, 'error_vs_information.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# =====================
# 7. Information Transmission Tracking
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

def plot_information_per_sync_event_hd(info_tracking, save_dir):
    """
    Plot high-definition line plot of information transmitted at each synchronization event.

    Args:
        info_tracking (dict): Information transmission data for each method
        save_dir (str): Directory to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Creating figure with high DPI
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    # Defining color palette
    colors = {
        'Adaptive Event': '#1f77b4',
        'MPC': '#ff7f0e',
        'Kalman': '#2ca02c',
        'RL': '#d62728'
    }

    # Defining line styles
    linestyles = {
        'Adaptive Event': '-',
        'MPC': '--',
        'Kalman': '-.',
        'RL': ':'
    }

    # Plotting each method with straight lines and no annotations
    for name, data in info_tracking.items():
        ax.plot(data['sync_times'], data['cumulative_bits'],
               label=name, color=colors[name],
               linestyle=linestyles[name],
               linewidth=2.5, alpha=0.9)

        ax.scatter(data['sync_times'], data['cumulative_bits'],
                  color=colors[name], s=80, zorder=3,
                  edgecolor='white', linewidth=1.5, alpha=0.8)

    # Adding title and labels with larger fonts
    ax.set_title('Information Transmitted at Each Synchronization Event',
                fontsize=18, fontweight='bold', color='#2c3e50', pad=25)
    ax.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Cumulative Bits Transmitted', fontsize=16, fontweight='bold', color='#2c3e50', labelpad=15)

    # Customizing grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)

    # Adding legend with better positioning
    legend = ax.legend(loc='upper left', fontsize=14, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    # Setting axis limits with padding
    max_time = max(max(data['sync_times'] for data in info_tracking.values()))
    max_bits = max(max(data['cumulative_bits'] for data in info_tracking.values()))
    ax.set_xlim(0, max_time * 1.05)
    ax.set_ylim(0, max_bits * 1.1)

    # Styling the plot for high definition
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Adjusting tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)

    # Setting face color
    ax.set_facecolor('#f8f9fa')

    # Using tight layout for better spacing
    plt.tight_layout()

    # Saving the plot
    plot_path = os.path.join(save_dir, 'information_transmission_hd.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# =====================
# 8. Main Execution
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
