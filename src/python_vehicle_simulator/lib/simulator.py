from tqdm import tqdm
import numpy as np, time
import matplotlib.pyplot as plt
from python_vehicle_simulator.lib.env import NavEnv
from typing import Literal, Tuple, List, Union
from copy import deepcopy

class Simulator:

    def __init__(
            self,
            env:NavEnv,
            *args,
            dt:float=None,
            skip_frames:int=0,
            render_mode:Literal['human']=None,
            verbose:int=0,
            window_size:Tuple=(10, 10),
            **kwargs
    ):
        self.dt = dt or env.dt
        self.env = env
        self.env.dt = self.dt
        self.skip_frames = skip_frames
        self.render_mode = render_mode
        self.verbose = verbose
        self.window_size = window_size
        
        # Storage for replay functionality
        self.simulation_data = {
            'timestamps': [],
            'own_vessel_states': [],
            'target_vessel_states': [],
            'obstacles': None,  # Static obstacles
            'gnc_data': {  # Store GNC "prev" data
                'navigation': [],
                'diagnosis': [],
                'guidance': [],
                'control': [],
                'actuators': []
            }
        }

    def run(self, tf:float, *args, render:bool=False, store_data:bool=True, **kwargs) -> None:
        """
        Run simulation from 0 to tf with sampling time self.dt
        """
        self.env.reset()
        print("Running simulation..")
        N = int(tf//self.dt) + 1
        
        # Store initial states if requested
        if store_data:
            self._clear_simulation_data()
            self.simulation_data['obstacles'] = deepcopy(self.env.obstacles)
            self._store_current_state()
        
        for t in tqdm(np.linspace(0, tf, N)):
            obs, r, term, trunc, info, done = self.env.step(*args, **kwargs)
            print([actuator.u_prev for actuator in self.env.own_vessel.actuators], self.env.own_vessel.nu.uvr)
            
            # Store state data for replay
            if store_data:
                self._store_current_state()
            
            if render and (self.skip_frames == 0 or (t//self.dt) % (self.skip_frames) == 0):
                self.env.render(self.render_mode, verbose=self.verbose, window_size=self.window_size)

    def _clear_simulation_data(self):
        """Clear stored simulation data"""
        self.simulation_data = {
            'timestamps': [],
            'own_vessel_states': [],
            'target_vessel_states': [],
            'obstacles': None,
            'gnc_data': {
                'navigation': [],
                'diagnosis': [],
                'guidance': [],
                'control': [],
                'actuators': []
            }
        }
    
    def _store_current_state(self):
        """Store current state of all vessels for replay"""
        # Store timestamp
        self.simulation_data['timestamps'].append(self.env.t)
        
        # Store own vessel state (eta and nu)
        own_vessel_state = {
            'eta': deepcopy(self.env.own_vessel.eta),
            'nu': deepcopy(self.env.own_vessel.nu)
        }
        self.simulation_data['own_vessel_states'].append(own_vessel_state)
        
        # Store target vessel states
        target_states = []
        for tv in self.env.target_vessels:
            target_state = {
                'eta': deepcopy(tv.eta),
                'nu': deepcopy(tv.nu)
            }
            target_states.append(target_state)
        self.simulation_data['target_vessel_states'].append(target_states)
        
        # Store GNC "prev" data from own vessel
        self._store_gnc_prev_data()
    
    def _store_gnc_prev_data(self):
        """Store the 'prev' data from guidance, navigation, control, diagnosis and actuators"""
        vessel = self.env.own_vessel
        
        # Store navigation prev data
        nav_prev = deepcopy(vessel.navigation.prev) if hasattr(vessel.navigation, 'prev') else None
        self.simulation_data['gnc_data']['navigation'].append(nav_prev)
        
        # Store diagnosis prev data
        diag_prev = deepcopy(vessel.diagnosis.prev) if hasattr(vessel.diagnosis, 'prev') else None
        self.simulation_data['gnc_data']['diagnosis'].append(diag_prev)
        
        # Store guidance prev data
        guid_prev = deepcopy(vessel.guidance.prev) if hasattr(vessel.guidance, 'prev') else None
        self.simulation_data['gnc_data']['guidance'].append(guid_prev)
        
        # Store control prev data
        ctrl_prev = deepcopy(vessel.control.prev) if hasattr(vessel.control, 'prev') else None
        self.simulation_data['gnc_data']['control'].append(ctrl_prev)
        
        # Store actuator prev data
        actuator_prev_data = []
        for actuator in vessel.actuators:
            act_prev = deepcopy(actuator.prev) if hasattr(actuator, 'prev') else None
            # print("Actuator: ", act_prev)
            actuator_prev_data.append(act_prev)
        self.simulation_data['gnc_data']['actuators'].append(actuator_prev_data)

    def replay(self, speed_factor:float=1.0, skip_frames:int=None) -> None:
        """
        Replay the stored simulation with visualization
        
        Args:
            speed_factor: Speed of replay (1.0 = real time, 2.0 = 2x speed, etc.)
            skip_frames: Number of frames to skip (default uses self.skip_frames)
        """
        if not self.simulation_data['timestamps']:
            print("No simulation data available. Run simulation first with store_data=True")
            return
        
        print("Replaying simulation...")
        skip = skip_frames if skip_frames is not None else self.skip_frames
        
        # Setup plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.ion()
        plt.show()
        
        # Get all vessel positions for setting axis limits
        all_positions = []
        for state in self.simulation_data['own_vessel_states']:
            all_positions.append([state['eta'][1], state['eta'][0]])  # [East, North]
        
        for target_states in self.simulation_data['target_vessel_states']:
            for state in target_states:
                all_positions.append([state['eta'][1], state['eta'][0]])
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 10
            x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
            y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        # Replay loop
        timestamps = self.simulation_data['timestamps']
        for i, t in enumerate(timestamps):
            if skip > 0 and i % (skip + 1) != 0:
                continue
                
            ax.cla()
            
            # Set vessel states for plotting
            own_state = self.simulation_data['own_vessel_states'][i]
            self.env.own_vessel.eta = own_state['eta']
            self.env.own_vessel.nu = own_state['nu']
            
            # Plot own vessel
            self.env.own_vessel.plot(ax=ax, verbose=self.verbose, c='blue', label='Own Vessel')
            
            # Plot target vessels
            target_states = self.simulation_data['target_vessel_states'][i]
            for j, target_state in enumerate(target_states):
                if j < len(self.env.target_vessels):
                    self.env.target_vessels[j].eta = target_state['eta']
                    self.env.target_vessels[j].nu = target_state['nu']
                    self.env.target_vessels[j].plot(ax=ax, verbose=self.verbose, c='red', 
                                                   label='Target Vessel' if j == 0 else '')
            
            # Plot obstacles
            if self.simulation_data['obstacles']:
                for obs in self.simulation_data['obstacles']:
                    obs.fill(ax=ax, color='grey', alpha=0.5, label='Obstacles')
                    obs.plot(ax=ax, color='black', alpha=0.8)
            
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.set_title(f"Simulation Replay (t={t:.1f}s)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Control replay speed
            if speed_factor > 0:
                time.sleep(self.dt / speed_factor)
        
        print("Replay completed!")

    def _extract_data_from_path(self, data_path):
        """
        Helper method to extract data from a given path notation.
        
        Parameters:
        -----------
        data_path : str
            Path in format 'component.attribute[index]' or 'component.attribute'
            
        Returns:
        --------
        tuple: (time_data, extracted_data, valid_indices, y_index_from_path)
        """
        # Parse the path
        path_parts = data_path.split('.')
        if len(path_parts) < 2:
            raise ValueError(f"Invalid data_path format: '{data_path}'. Expected format: 'component.key1.key2...'")
        
        component = path_parts[0]
        attribute_path = path_parts[1:]
        
        # Handle actuator indexing
        actuator_index = None
        if component.startswith('actuators[') and component.endswith(']'):
            try:
                actuator_index = int(component[10:-1])
                component = 'actuators'
            except ValueError:
                raise ValueError(f"Invalid actuator index in '{component}'")
        
        # Handle index notation in attribute path (e.g., "navigation.eta[0]")
        y_index_from_path = None
        if len(attribute_path) > 0:
            last_attr = attribute_path[-1]
            if '[' in last_attr and last_attr.endswith(']'):
                base_attr, index_part = last_attr.rsplit('[', 1)
                y_index_from_path = int(index_part[:-1])
                attribute_path[-1] = base_attr
        
        # Get time and component data
        time_data = np.array(self.simulation_data['timestamps'])
        
        if component == 'vessel':
            component_data = self.simulation_data['own_vessel_states']
        else:
            component_data = self.simulation_data['gnc_data'][component]
        
        # Extract time series data
        extracted_data = []
        for timestep_data in component_data:
            if timestep_data is None:
                extracted_data.append(None)
                continue
                
            try:
                current_data = timestep_data
                
                # Handle vessel state data
                if component == 'vessel':
                    # Navigate through the attribute path (e.g., ['eta'] or ['nu'])
                    for key in attribute_path:
                        if isinstance(current_data, dict) and key in current_data:
                            current_data = current_data[key]
                            # Convert Eta/Nu objects to numpy arrays
                            if hasattr(current_data, 'to_numpy'):
                                current_data = current_data.to_numpy()
                        else:
                            current_data = None
                            break
                    extracted_data.append(current_data)
                    continue
                
                # Handle actuators separately (it's a list of actuator data)
                if component == 'actuators':
                    if isinstance(timestep_data, list) and len(timestep_data) > 0:
                        # Check if specific actuator index was requested
                        if actuator_index is not None:
                            if actuator_index < len(timestep_data):
                                current_data = timestep_data[actuator_index]
                            else:
                                # Requested actuator index doesn't exist for this timestep
                                current_data = None
                        else:
                            # Default: use the first actuator's data
                            current_data = timestep_data[0]
                        
                        if current_data is None:
                            extracted_data.append(None)
                            continue
                    else:
                        extracted_data.append(None)
                        continue
                
                # Navigate through the attribute path
                for key in attribute_path:
                    if isinstance(current_data, dict) and key in current_data:
                        current_data = current_data[key]
                    else:
                        current_data = None
                        break
                
                extracted_data.append(current_data)
                
            except (KeyError, TypeError, AttributeError):
                extracted_data.append(None)
        
        # Filter out None values and corresponding time points
        valid_indices = [i for i, data in enumerate(extracted_data) if data is not None]
        
        if not valid_indices:
            raise ValueError(f"No valid data found for '{data_path}'. Check if the component generates this data.")
        
        return time_data, extracted_data, valid_indices, y_index_from_path

    def plot_gnc_data(self, data_path: str, y_indices: List[int] = None, x_path: str = None, 
                      fig_size: Tuple[int, int] = (12, 8), title: str = None, 
                      ylabel: str = None, xlabel: str = None, legend_labels: List[str] = None,
                      colors: List[str] = None, linestyles: List[str] = None) -> plt.Figure:
        """
        Plot GNC data from the stored simulation data with support for deep dot notation.
        
        Args:
            data_path: String with dot notation (e.g., "navigation.eta", "guidance.eta_des")
                      For actuators, you can specify which actuator: "actuators[0].tau", "actuators[1].info.u_actual"
                      For vessel states, use: "vessel.eta", "vessel.nu"
            y_indices: List of indices to plot as y coordinates. If None, treats data as scalar
            x_path: String with dot notation for x coordinate (e.g., "navigation.eta[0]"). If None, uses time
            fig_size: Figure size tuple (width, height)
            title: Plot title. If None, auto-generated from data_path
            ylabel: Y-axis label. If None, auto-generated
            xlabel: X-axis label. If None, uses "Time (s)" or auto-generated
            legend_labels: Custom legend labels. If None, auto-generated
            colors: List of colors for each y-series
            linestyles: List of linestyles for each y-series
            
        Returns:
            matplotlib Figure object
            
        Example:
            # Plot vessel position and orientation
            sim.plot_gnc_data("vessel.eta", y_indices=[0, 1])  # North and East position
            sim.plot_gnc_data("vessel.eta", y_indices=[5])     # Heading
            
            # Plot vessel velocities
            sim.plot_gnc_data("vessel.nu", y_indices=[0, 1, 5])  # Surge, sway, yaw rate
            
            # Plot vessel position from navigation data
            sim.plot_gnc_data("navigation.eta", y_indices=[0, 1])
            
            # Plot guidance desired heading
            sim.plot_gnc_data("guidance.eta_des", y_indices=[5])
            
            # Plot first actuator's forces
            sim.plot_gnc_data("actuators[0].tau", y_indices=[0, 2])
            
            # Plot second actuator's actual inputs
            sim.plot_gnc_data("actuators[1].info.u_actual", y_indices=[0])
            
            # Plot surge velocity vs time
            sim.plot_gnc_data("navigation.nu", y_indices=[0])
            
            # Plot north vs east position (trajectory)
            sim.plot_gnc_data("vessel.eta", y_indices=[0], x_path="vessel.eta[1]")
            
            # Plot east vs north position (east-north trajectory)
            sim.plot_gnc_data("vessel.eta[1]", x_path="vessel.eta[0]")
        """
        
        if not self.simulation_data['timestamps']:
            raise ValueError("No simulation data available. Run simulation first with store_data=True")
        
        # Use helper method to extract data
        time_data, extracted_data, valid_indices, y_index_from_path = self._extract_data_from_path(data_path)
        
        valid_time = time_data[valid_indices]
        valid_data = [extracted_data[i] for i in valid_indices]
        
        # Convert to numpy array and handle different data types
        try:
            data_array = np.array(valid_data)
        except:
            # Handle case where data elements have different shapes
            max_len = max(len(d) if hasattr(d, '__len__') and not isinstance(d, str) else 1 for d in valid_data)
            data_array = np.full((len(valid_data), max_len), np.nan)
            for i, d in enumerate(valid_data):
                if hasattr(d, '__len__') and not isinstance(d, str):
                    data_array[i, :len(d)] = d
                else:
                    data_array[i, 0] = d
        
        # Handle scalar vs vector data and index extraction
        if data_array.ndim == 1:
            # Scalar data - can't use indices
            if y_indices is not None:
                print(f"Warning: y_indices specified but data is scalar. Ignoring y_indices.")
            if y_index_from_path is not None:
                print(f"Warning: index notation [{y_index_from_path}] specified but data is scalar. Ignoring index.")
            y_data = [data_array]
            data_labels = [data_path]
        else:
            # Vector data - determine which indices to plot
            if y_indices is not None:
                # Explicit y_indices parameter takes precedence
                indices_to_plot = y_indices
            elif y_index_from_path is not None:
                # Use index from path notation (e.g., "vessel.eta[1]")
                indices_to_plot = [y_index_from_path]
            else:
                # Default: plot all components
                indices_to_plot = list(range(data_array.shape[1]))
            
            # Validate indices
            for idx in indices_to_plot:
                if idx >= data_array.shape[1]:
                    raise ValueError(f"Index {idx} is out of bounds for data with {data_array.shape[1]} components")
            
            y_data = [data_array[:, idx] for idx in indices_to_plot]
            
            # Generate appropriate labels
            if y_index_from_path is not None and y_indices is None:
                # For path notation like "vessel.eta[1]", use the original path as label
                data_labels = [data_path]
            else:
                # For explicit y_indices or default case, use indexed labels
                component = data_path.split('.')[0]
                attribute_path = data_path.split('.')[1:]
                data_labels = [f"{'.'.join([component] + attribute_path)}[{idx}]" for idx in indices_to_plot]
        
        # Determine x-axis data
        if x_path is not None:
            # Extract x-axis data using helper method
            x_time_data, x_extracted_data, x_valid_indices, x_index_from_path = self._extract_data_from_path(x_path)
            
            # Filter x_data to match y_data indices
            x_valid_data = [x_extracted_data[i] for i in valid_indices]
            
            if not all(d is not None for d in x_valid_data):
                raise ValueError(f"Some x-axis data points are None for '{x_path}'")
            
            # Convert to numpy array and extract index if specified
            try:
                x_data_array = np.array(x_valid_data)
            except:
                raise ValueError(f"Could not convert x-axis data to numpy array for '{x_path}'")
            
            # Handle index extraction for x-data
            if x_index_from_path is not None:
                if x_data_array.ndim == 1:
                    raise ValueError(f"Cannot use index notation with scalar x-data: '{x_path}'")
                if x_index_from_path >= x_data_array.shape[1]:
                    raise ValueError(f"x_path index {x_index_from_path} is out of bounds for data with {x_data_array.shape[1]} components")
                x_data = x_data_array[:, x_index_from_path]
            else:
                if x_data_array.ndim > 1:
                    # Default to first component if no index specified
                    x_data = x_data_array[:, 0]
                else:
                    x_data = x_data_array
            
            x_label = xlabel or x_path
        else:
            x_data = valid_time
            x_label = xlabel or "Time (s)"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Set up colors and linestyles
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(y_data)))
        if linestyles is None:
            linestyles = ['-'] * len(y_data)
        
        # Plot each y-series
        for i, (y_series, label) in enumerate(zip(y_data, data_labels)):
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            legend_label = legend_labels[i] if legend_labels and i < len(legend_labels) else label
            
            ax.plot(x_data, y_series, color=color, linestyle=linestyle, 
                   linewidth=2, label=legend_label, marker='o', markersize=3, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or data_path, fontsize=12, fontweight='bold')
        ax.set_title(title or f"GNC Data: {data_path}", fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend if multiple series
        if len(y_data) > 1:
            ax.legend(fontsize=10, framealpha=0.9)
        
        # Improve layout
        plt.tight_layout()
        
        # For actuator data, ensure y-axis includes 0 (important reference for forces/moments)
        component = data_path.split('.')[0]
        if component == 'actuators':
            y_min, y_max = ax.get_ylim()
            if y_min > 0:
                ax.set_ylim(bottom=0)
            elif y_max < 0:
                ax.set_ylim(top=0)
        
        # Add some styling for scientific presentation
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        return fig

    def plot_gnc_data_multi(self, data_paths: List[str], labels: List[str] = None, 
                           colors: List[str] = None, linestyles: List[str] = None,
                           fig_size: Tuple[int, int] = (12, 8), title: str = None, 
                           ylabel: str = None, xlabel: str = None, 
                           x_path: Union[str, List[Union[str, None]], None] = None) -> plt.Figure:
        """
        Plot multiple GNC data paths on the same figure for comparison.
        
        Args:
            data_paths: List of data path strings with optional index notation
                       e.g., ["navigation.eta[0]", "guidance.eta_des[0]"]
                       If no [index] specified, treats as scalar data
            labels: Custom labels for each data path. If None, auto-generated
            colors: List of colors for each data path. If None, uses default color cycle
            linestyles: List of linestyles for each data path. If None, uses solid lines
            fig_size: Figure size tuple (width, height)
            title: Plot title. If None, auto-generated
            ylabel: Y-axis label. If None, auto-generated
            xlabel: X-axis label. If None, uses "Time (s)" or auto-generated
            x_path: X-coordinate specification. Can be:
                   - None: Use time as x-axis for all series
                   - str: Single x_path used for all series (e.g., "navigation.eta[0]")
                   - List[str]: Different x_path for each series (e.g., ["vessel.eta[0]", "navigation.eta[0]"])
                   - List with None: Mix of time and custom x-coordinates (e.g., [None, "vessel.eta[0]"])
            
        Returns:
            matplotlib Figure object
            
        Example:
            # Compare vessel actual vs navigation estimated position
            sim.plot_gnc_data_multi([
                "vessel.eta[0]",                    # Actual north position
                "navigation.eta[0]"                 # Navigation estimated north
            ], labels=["Vessel Actual", "Navigation Est"])
            
            # Compare actual vs desired position
            sim.plot_gnc_data_multi([
                "vessel.eta[0]",                    # Actual north position
                "guidance.eta_des[0]"               # Desired north position
            ], labels=["Actual North", "Desired North"])
            
            # Plot east vs north position from both navigation and vessel (single x_path)
            sim.plot_gnc_data_multi([
                "navigation.eta[1]",                # Navigation east position
                "vessel.eta[1]"                     # Vessel actual east position
            ], x_path="navigation.eta[0]",          # Use north position as x-axis for both
            labels=["Navigation East vs North", "Vessel Actual East vs North"],
            xlabel="North Position (m)", ylabel="East Position (m)")
            
            # Plot different trajectories with different coordinate systems (multiple x_path)
            sim.plot_gnc_data_multi([
                "navigation.eta[1]",                # Navigation east position
                "vessel.eta[1]"                     # Vessel actual east position
            ], x_path=["navigation.eta[0]", "vessel.eta[0]"],  # Different x-coordinates
            labels=["Navigation Trajectory", "Vessel Trajectory"],
            xlabel="North Position (m)", ylabel="East Position (m)")
            
            # Mix time-based and trajectory plotting
            sim.plot_gnc_data_multi([
                "vessel.eta[0]",                    # North vs time
                "vessel.eta[1]"                     # East vs north
            ], x_path=[None, "vessel.eta[0]"],      # Time for first, north for second
            labels=["North vs Time", "East vs North"])
            
            # Plot vessel trajectory components
            sim.plot_gnc_data_multi([
                "vessel.eta[0]",                    # North position
                "vessel.eta[1]",                    # East position
                "vessel.eta[5]"                     # Heading
            ], labels=["North", "East", "Heading"])
            
            # Plot vessel trajectory components
            sim.plot_gnc_data_multi([
                "vessel.eta[0]",                    # North position
                "vessel.eta[1]",                    # East position
                "vessel.eta[5]"                     # Heading
            ], labels=["North", "East", "Heading"])
            
            # Compare multiple actuator forces
            sim.plot_gnc_data_multi([
                "actuators[0].tau[0]",
                "actuators[1].tau[0]",
                "actuators[2].tau[0]"
            ], labels=["Actuator 1", "Actuator 2", "Actuator 3"])
            
            # Mix vessel states with control data
            sim.plot_gnc_data_multi([
                "vessel.nu[0]",                     # Actual surge velocity
                "guidance.nu_des[0]",               # Desired surge velocity
                "control.u[0]"                      # Control force
            ], labels=["Actual Surge", "Desired Surge", "Control Force"])
        """
        
        if not self.simulation_data['timestamps']:
            raise ValueError("No simulation data available. Run simulation first with store_data=True")
        
        if not data_paths:
            raise ValueError("At least one data path must be provided")
        
        # Extract data for each path using the helper method
        all_series_data = []
        all_series_labels = []
        has_actuator_data = False
        time_data = None
        valid_indices = None
        
        for i, path in enumerate(data_paths):
            try:
                # Parse path with optional index notation: "component.attr[index]"
                y_index_from_path = None
                if '[' in path and path.endswith(']'):
                    # Path already has index notation, use as-is
                    data_path = path
                else:
                    # No index specified, use path as-is (helper will handle scalar/vector appropriately)
                    data_path = path
                
                # Check if this is actuator data
                if data_path.startswith('actuators'):
                    has_actuator_data = True
                
                # Extract data using helper method
                path_time_data, extracted_data, path_valid_indices, path_y_index = self._extract_data_from_path(data_path)
                
                # Use first path's time and valid indices as reference
                if time_data is None:
                    time_data = path_time_data
                    valid_indices = path_valid_indices
                
                # Get valid data
                valid_data = [extracted_data[idx] for idx in path_valid_indices]
                
                # Convert to numpy array
                try:
                    data_array = np.array(valid_data)
                except:
                    # Handle different shapes
                    max_len = max(len(d) if hasattr(d, '__len__') and not isinstance(d, str) else 1 for d in valid_data)
                    data_array = np.full((len(valid_data), max_len), np.nan)
                    for j, d in enumerate(valid_data):
                        if hasattr(d, '__len__') and not isinstance(d, str):
                            data_array[j, :len(d)] = d
                        else:
                            data_array[j, 0] = d
                
                # Extract the specific component based on index notation
                if data_array.ndim == 1:
                    # Scalar data
                    if path_y_index is not None:
                        print(f"Warning: index notation in '{path}' ignored for scalar data")
                    y_series = data_array
                else:
                    # Vector data
                    if path_y_index is not None:
                        # Use index from path notation
                        if path_y_index >= data_array.shape[1]:
                            raise ValueError(f"Index {path_y_index} out of bounds for '{path}' with {data_array.shape[1]} components")
                        y_series = data_array[:, path_y_index]
                    else:
                        # No index specified, use first component as default
                        y_series = data_array[:, 0]
                
                all_series_data.append(y_series)
                
                # Generate label
                if labels and i < len(labels):
                    all_series_labels.append(labels[i])
                else:
                    all_series_labels.append(path)
                    
            except Exception as e:
                print(f"Warning: Failed to extract data for '{path}': {e}")
                continue
        
        if not all_series_data:
            raise ValueError("No valid data found for any of the provided paths")
        
        # Determine x-axis data - support single x_path, list of x_paths, or None
        x_data_list = []
        x_label = xlabel or "Time (s)"
        
        # Normalize x_path to a list
        if x_path is None:
            # Use time for all series
            x_paths = [None] * len(all_series_data)
        elif isinstance(x_path, str):
            # Single x_path for all series
            x_paths = [x_path] * len(all_series_data)
        elif isinstance(x_path, list):
            # List of x_paths, one for each series
            if len(x_path) != len(all_series_data):
                raise ValueError(f"Number of x_paths ({len(x_path)}) must match number of data_paths ({len(all_series_data)})")
            x_paths = x_path
        else:
            raise ValueError("x_path must be None, str, or List[str]")
        
        # Extract x-data for each series
        for i, (y_series, current_x_path) in enumerate(zip(all_series_data, x_paths)):
            if current_x_path is None:
                # Use time for this series
                x_data_list.append(time_data[valid_indices])
                if isinstance(x_path, list) and any(xp is not None for xp in x_path):
                    # Mixed x-coordinates, keep specific label
                    if xlabel is None:
                        x_label = "Mixed Coordinates"
            else:
                try:
                    # Extract x-axis data using helper method
                    x_time_data, x_extracted_data, x_path_valid_indices, x_index_from_path = self._extract_data_from_path(current_x_path)
                    
                    # Filter x_data to match our reference valid_indices
                    x_valid_data = [x_extracted_data[idx] for idx in valid_indices]
                    
                    if not all(d is not None for d in x_valid_data):
                        raise ValueError(f"Some x-axis data points are None for '{current_x_path}'")
                    
                    # Convert to numpy array
                    try:
                        x_data_array = np.array(x_valid_data)
                    except:
                        raise ValueError(f"Could not convert x-axis data to numpy array for '{current_x_path}'")
                    
                    # Handle index extraction for x-data
                    if x_index_from_path is not None:
                        if x_data_array.ndim == 1:
                            raise ValueError(f"Cannot use index notation with scalar x-data: '{current_x_path}'")
                        if x_index_from_path >= x_data_array.shape[1]:
                            raise ValueError(f"x_path index {x_index_from_path} out of bounds for data with {x_data_array.shape[1]} components")
                        current_x_data = x_data_array[:, x_index_from_path]
                    else:
                        if x_data_array.ndim > 1:
                            # Default to first component
                            current_x_data = x_data_array[:, 0]
                        else:
                            current_x_data = x_data_array
                    
                    x_data_list.append(current_x_data)
                    
                    # Set x_label if not manually specified
                    if xlabel is None:
                        if isinstance(x_path, str):
                            # Single x_path for all series
                            x_label = x_path
                        elif isinstance(x_path, list) and len(set(x_path)) == 1:
                            # All series use same x_path
                            x_label = current_x_path
                        else:
                            # Multiple different x_paths
                            x_label = "Mixed Coordinates"
                    
                except Exception as e:
                    print(f"Warning: Failed to extract x-axis data from '{current_x_path}' for series {i}: {e}")
                    print("Falling back to time as x-axis for this series")
                    x_data_list.append(time_data[valid_indices])
        
        # Create the comparison plot
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Set up colors and linestyles
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_series_data)))
        if linestyles is None:
            linestyles = ['-'] * len(all_series_data)
        
        # Plot each series
        for i, (y_data, label, x_data) in enumerate(zip(all_series_data, all_series_labels, x_data_list)):
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            # Ensure x and y data have same length
            min_len = min(len(x_data), len(y_data))
            x_plot = x_data[:min_len]
            y_plot = y_data[:min_len]
            
            ax.plot(x_plot, y_plot, color=color, linestyle=linestyle, 
                   linewidth=2, label=label, marker='o', markersize=3, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or "Value", fontsize=12, fontweight='bold')
        ax.set_title(title or "GNC Data Comparison", fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(fontsize=10, framealpha=0.9)
        
        # Improve layout
        plt.tight_layout()
        
        # For actuator data, ensure y-axis includes 0
        if has_actuator_data:
            y_min, y_max = ax.get_ylim()
            if y_min > 0:
                ax.set_ylim(bottom=0)
            elif y_max < 0:
                ax.set_ylim(top=0)
        
        # Add some styling for scientific presentation
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        return fig

    def save_animation(self, filename: str = "simulation_animation", format: str = "gif", 
                   fps: int = 10, duration_per_frame: float = None, dpi: int = 100,
                   skip_frames: int = None, figsize: Tuple[int, int] = (10, 8)) -> str:
        """
        Generate and save an animation (GIF or MP4) from the stored simulation data.
        
        Args:
            filename: Output filename (without extension)
            format: Animation format - "gif" or "mp4"
            fps: Frames per second for the animation
            duration_per_frame: Duration per frame in seconds (for GIF). If None, calculated from fps
            dpi: Resolution of the animation
            skip_frames: Number of frames to skip (default uses self.skip_frames)
            figsize: Figure size tuple (width, height)
            
        Returns:
            str: Path to the saved animation file
            
        Example:
            # Generate GIF at 15 fps
            sim.save_animation("my_simulation", format="gif", fps=15)
            
            # Generate MP4 video at 30 fps with higher resolution
            sim.save_animation("vessel_tracking", format="mp4", fps=30, dpi=150)
            
            # Generate GIF with custom frame duration
            sim.save_animation("slow_motion", format="gif", duration_per_frame=0.2)
        """
    
        if not self.simulation_data['timestamps']:
            raise ValueError("No simulation data available. Run simulation first with store_data=True")
        
        # Import animation libraries
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
            import os
        except ImportError as e:
            raise ImportError(f"Required animation libraries not available: {e}. "
                            "Install with: pip install matplotlib[animation] pillow")
        
        if format.lower() == "mp4":
            try:
                # Test if ffmpeg is available
                import subprocess
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("FFmpeg not found. Install FFmpeg for MP4 support or use GIF format instead.")
        
        print(f"Generating {format.upper()} animation...")
        
        # Setup parameters
        skip = skip_frames if skip_frames is not None else self.skip_frames
        timestamps = self.simulation_data['timestamps']
        
        # Filter frames based on skip_frames
        if skip > 0:
            frame_indices = list(range(0, len(timestamps), skip + 1))
        else:
            frame_indices = list(range(len(timestamps)))
        
        # Setup figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Calculate axis limits from all positions
        all_positions = []
        for state in self.simulation_data['own_vessel_states']:
            all_positions.append([state['eta'][1], state['eta'][0]])  # [East, North]
        
        for target_states in self.simulation_data['target_vessel_states']:
            for state in target_states:
                all_positions.append([state['eta'][1], state['eta'][0]])
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 10
            x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
            y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        def animate_frame(frame_num):
            """Animation function called for each frame"""
            i = frame_indices[frame_num]
            t = timestamps[i]
            
            ax.clear()
            
            # Set vessel states for plotting
            own_state = self.simulation_data['own_vessel_states'][i]
            self.env.own_vessel.eta = own_state['eta']
            self.env.own_vessel.nu = own_state['nu']
            
            # Plot own vessel
            self.env.own_vessel.plot(ax=ax, verbose=self.verbose, c='blue', label='Own Vessel')
            
            # Plot target vessels
            target_states = self.simulation_data['target_vessel_states'][i]
            for j, target_state in enumerate(target_states):
                if j < len(self.env.target_vessels):
                    self.env.target_vessels[j].eta = target_state['eta']
                    self.env.target_vessels[j].nu = target_state['nu']
                    self.env.target_vessels[j].plot(ax=ax, verbose=self.verbose, c='red', 
                                                label='Target Vessel' if j == 0 else '')
            
            # Plot obstacles
            if self.simulation_data['obstacles']:
                for obs in self.simulation_data['obstacles']:
                    obs.fill(ax=ax, color='grey', alpha=0.5, label='Obstacles')
                    obs.plot(ax=ax, color='black', alpha=0.8)
            
            # Plot trajectory trail (optional - shows path taken)
            if frame_num > 0:
                # Plot trail for own vessel
                trail_indices = frame_indices[:frame_num+1]
                trail_east = [self.simulation_data['own_vessel_states'][idx]['eta'][1] for idx in trail_indices]
                trail_north = [self.simulation_data['own_vessel_states'][idx]['eta'][0] for idx in trail_indices]
                ax.plot(trail_east, trail_north, 'b--', alpha=0.6, linewidth=1, label='Own Vessel Trail')
                
                # Plot trail for target vessels
                for j, _ in enumerate(self.env.target_vessels):
                    if j < len(self.env.target_vessels):
                        trail_east = [self.simulation_data['target_vessel_states'][idx][j]['eta'][1] 
                                    for idx in trail_indices if j < len(self.simulation_data['target_vessel_states'][idx])]
                        trail_north = [self.simulation_data['target_vessel_states'][idx][j]['eta'][0] 
                                    for idx in trail_indices if j < len(self.simulation_data['target_vessel_states'][idx])]
                        if trail_east and trail_north:
                            ax.plot(trail_east, trail_north, 'r--', alpha=0.6, linewidth=1, 
                                label='Target Trail' if j == 0 else '')
            
            # Set plot properties
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_xlabel('East (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('North (m)', fontsize=12, fontweight='bold')
            ax.set_title(f"Vessel Simulation (t={t:.1f}s)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            # ax.legend(fontsize=10, loc='upper right')
            ax.set_aspect('equal')
            
            # Add styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        
        # Create animation
        total_frames = len(frame_indices)
        anim = FuncAnimation(fig, animate_frame, frames=total_frames, interval=1000/fps, blit=False, repeat=True)
        
        # Save animation
        output_file = f"{filename}.{format.lower()}"
        
        if format.lower() == "gif":
            # Calculate duration per frame for GIF
            if duration_per_frame is None:
                duration_per_frame = 1000 / fps  # Convert fps to milliseconds per frame
            else:
                duration_per_frame = duration_per_frame * 1000  # Convert seconds to milliseconds
            
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
            
        elif format.lower() == "mp4":
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_file, writer=writer, dpi=dpi)
            
        else:
            raise ValueError(f"Unsupported format '{format}'. Supported formats: 'gif', 'mp4'")
        
        plt.close(fig)  # Clean up
        
        print(f"Animation saved as: {output_file}")
        print(f"Total frames: {total_frames}, Duration: {total_frames/fps:.1f} seconds")
        
        return output_file

    def save_animation_with_trails(self, filename: str = "simulation_with_trails", format: str = "gif", 
                                fps: int = 10, trail_length: int = 50, dpi: int = 100,
                                skip_frames: int = None, figsize: Tuple[int, int] = (10, 8)) -> str:
        """
        Generate animation with dynamic trailing paths that show recent vessel movements.
        
        Args:
            filename: Output filename (without extension)  
            format: Animation format - "gif" or "mp4"
            fps: Frames per second for the animation
            trail_length: Number of recent positions to show in trail
            dpi: Resolution of the animation
            skip_frames: Number of frames to skip (default uses self.skip_frames)
            figsize: Figure size tuple (width, height)
            
        Returns:
            str: Path to the saved animation file
            
        Example:
            # Generate GIF with 30-point trailing paths
            sim.save_animation_with_trails("vessel_trails", trail_length=30, fps=15)
        """
        
        if not self.simulation_data['timestamps']:
            raise ValueError("No simulation data available. Run simulation first with store_data=True")
        
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
            import os
        except ImportError as e:
            raise ImportError(f"Required animation libraries not available: {e}")
        
        if format.lower() == "mp4":
            try:
                import subprocess
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("FFmpeg not found. Install FFmpeg for MP4 support or use GIF format instead.")
        
        print(f"Generating {format.upper()} animation with dynamic trails...")
        
        # Setup parameters  
        skip = skip_frames if skip_frames is not None else self.skip_frames
        timestamps = self.simulation_data['timestamps']
        
        if skip > 0:
            frame_indices = list(range(0, len(timestamps), skip + 1))
        else:
            frame_indices = list(range(len(timestamps)))
        
        # Setup figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Calculate axis limits
        all_positions = []
        for state in self.simulation_data['own_vessel_states']:
            all_positions.append([state['eta'][1], state['eta'][0]])
        
        for target_states in self.simulation_data['target_vessel_states']:
            for state in target_states:
                all_positions.append([state['eta'][1], state['eta'][0]])
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 10
            x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
            y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        def animate_frame_with_trails(frame_num):
            """Animation function with dynamic trails"""
            i = frame_indices[frame_num]
            t = timestamps[i]
            
            ax.clear()
            
            # Calculate trail range
            trail_start = max(0, frame_num - trail_length)
            trail_indices = frame_indices[trail_start:frame_num+1]
            
            # Plot dynamic trails first (so vessels appear on top)
            if len(trail_indices) > 1:
                # Own vessel trail
                trail_east = [self.simulation_data['own_vessel_states'][idx]['eta'][1] for idx in trail_indices]
                trail_north = [self.simulation_data['own_vessel_states'][idx]['eta'][0] for idx in trail_indices]
                
                # Create fading effect for trail
                alphas = np.linspace(0.1, 0.8, len(trail_east))
                for j in range(len(trail_east)-1):
                    ax.plot(trail_east[j:j+2], trail_north[j:j+2], 'b-', alpha=alphas[j], linewidth=2)
                
                # Target vessel trails
                for vessel_idx, _ in enumerate(self.env.target_vessels):
                    trail_east = [self.simulation_data['target_vessel_states'][idx][vessel_idx]['eta'][1] 
                                for idx in trail_indices if vessel_idx < len(self.simulation_data['target_vessel_states'][idx])]
                    trail_north = [self.simulation_data['target_vessel_states'][idx][vessel_idx]['eta'][0] 
                                for idx in trail_indices if vessel_idx < len(self.simulation_data['target_vessel_states'][idx])]
                    
                    if len(trail_east) > 1:
                        alphas = np.linspace(0.1, 0.8, len(trail_east))
                        for j in range(len(trail_east)-1):
                            ax.plot(trail_east[j:j+2], trail_north[j:j+2], 'r-', alpha=alphas[j], linewidth=2)
            
            # Set current vessel states and plot vessels
            own_state = self.simulation_data['own_vessel_states'][i]
            self.env.own_vessel.eta = own_state['eta']
            self.env.own_vessel.nu = own_state['nu']
            self.env.own_vessel.plot(ax=ax, verbose=self.verbose, c='blue', label='Own Vessel')
            
            # Plot target vessels
            target_states = self.simulation_data['target_vessel_states'][i]
            for j, target_state in enumerate(target_states):
                if j < len(self.env.target_vessels):
                    self.env.target_vessels[j].eta = target_state['eta']
                    self.env.target_vessels[j].nu = target_state['nu']
                    self.env.target_vessels[j].plot(ax=ax, verbose=self.verbose, c='red', 
                                                label='Target Vessel' if j == 0 else '')
            
            # Plot obstacles
            if self.simulation_data['obstacles']:
                for obs in self.simulation_data['obstacles']:
                    obs.fill(ax=ax, color='grey', alpha=0.5, label='Obstacles')
                    obs.plot(ax=ax, color='black', alpha=0.8)
            
            # Set plot properties
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_xlabel('East (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('North (m)', fontsize=12, fontweight='bold')
            ax.set_title(f"Vessel Simulation with Trails (t={t:.1f}s)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            # ax.legend(fontsize=10, loc='upper right')
            ax.set_aspect('equal')
            
            # Add styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        
        # Create animation
        total_frames = len(frame_indices)
        anim = FuncAnimation(fig, animate_frame_with_trails, frames=total_frames, interval=1000/fps, blit=False, repeat=True)
        
        # Save animation
        output_file = f"{filename}.{format.lower()}"
        
        if format.lower() == "gif":
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
        elif format.lower() == "mp4":
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_file, writer=writer, dpi=dpi)
        else:
            raise ValueError(f"Unsupported format '{format}'. Supported formats: 'gif', 'mp4'")
        
        plt.close(fig)
        
        print(f"Animation with trails saved as: {output_file}")
        print(f"Total frames: {total_frames}, Duration: {total_frames/fps:.1f} seconds, Trail length: {trail_length}")
        
        return output_file

