import tkinter as tk
from tkinter import ttk, messagebox
import generate_synthetic_data as gsd
from functools import partial
import threading
import os
import pyttsx3

class SyntheticDataGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Medical Data Generator")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scenarios selection
        self.create_scenarios_frame(main_frame)
        
        # Data types selection
        self.create_data_types_frame(main_frame)
        
        # Number of instances
        self.create_instances_frame(main_frame)
        
        # Progress bar
        self.create_progress_frame(main_frame)
        
        # Generate button
        self.generate_btn = ttk.Button(
            main_frame, 
            text="Generate Data", 
            command=self.start_generation
        )
        self.generate_btn.grid(row=4, column=0, pady=10, sticky=tk.EW)

    def create_scenarios_frame(self, parent):
        # Scenarios Frame
        scenarios_frame = ttk.LabelFrame(parent, text="Select Scenarios", padding="5")
        scenarios_frame.grid(row=0, column=0, sticky=tk.NSEW, pady=5)
        
        # Select All button for scenarios
        self.all_scenarios_var = tk.BooleanVar()
        ttk.Checkbutton(
            scenarios_frame,
            text="Select All Scenarios",
            variable=self.all_scenarios_var,
            command=self.toggle_all_scenarios
        ).grid(row=0, column=0, sticky=tk.W)
        
        # Scrollable frame for scenarios
        canvas = tk.Canvas(scenarios_frame, height=200)
        scrollbar = ttk.Scrollbar(scenarios_frame, orient="vertical", command=canvas.yview)
        self.scenarios_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.grid(row=1, column=1, sticky=tk.NS)
        canvas.grid(row=1, column=0, sticky=tk.NSEW)
        
        # Create window in canvas
        canvas.create_window((0, 0), window=self.scenarios_frame, anchor=tk.NW)
        
        # Scenario checkbuttons
        self.scenario_vars = {}
        for i, scenario in enumerate(gsd.SCENARIOS):
            var = tk.BooleanVar()
            self.scenario_vars[scenario['id']] = var
            ttk.Checkbutton(
                self.scenarios_frame,
                text=f"{scenario['name']} (Emergency: {scenario['emergency']})",
                variable=var
            ).grid(row=i, column=0, sticky=tk.W)
        
        # Update scroll region
        self.scenarios_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def create_data_types_frame(self, parent):
        # Data Types Frame
        data_types_frame = ttk.LabelFrame(parent, text="Select Data Types", padding="5")
        data_types_frame.grid(row=1, column=0, sticky=tk.NSEW, pady=5)
        
        # Data types checkbuttons
        self.data_type_vars = {}
        data_types = {
            'text': 'Text data (descriptions and transcripts)',
            'image': 'Image data (facial expressions)',
            'audio': 'Audio data (synthesized speech)',
            'video': 'Video data (simple animations)',
            'physiological': 'Physiological data (vital signs)',
            'database': 'Database entries (metadata)'
        }
        
        # Select All button for data types
        self.all_types_var = tk.BooleanVar()
        ttk.Checkbutton(
            data_types_frame,
            text="Select All Data Types",
            variable=self.all_types_var,
            command=self.toggle_all_data_types
        ).grid(row=0, column=0, sticky=tk.W)
        
        for i, (key, description) in enumerate(data_types.items(), 1):
            var = tk.BooleanVar()
            self.data_type_vars[key] = var
            ttk.Checkbutton(
                data_types_frame,
                text=description,
                variable=var
            ).grid(row=i, column=0, sticky=tk.W)

    def create_instances_frame(self, parent):
        # Instances Frame
        instances_frame = ttk.LabelFrame(parent, text="Number of Instances", padding="5")
        instances_frame.grid(row=2, column=0, sticky=tk.NSEW, pady=5)
        
        self.instances_var = tk.StringVar(value="100")
        ttk.Entry(
            instances_frame,
            textvariable=self.instances_var,
            width=10
        ).grid(row=0, column=0, padx=5)
        
        ttk.Label(
            instances_frame,
            text="instances per scenario"
        ).grid(row=0, column=1, padx=5)

    def create_progress_frame(self, parent):
        # Progress Frame
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, sticky=tk.NSEW, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky=tk.EW)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            progress_frame,
            textvariable=self.status_var
        )
        self.status_label.grid(row=1, column=0, sticky=tk.W)

    def toggle_all_scenarios(self):
        state = self.all_scenarios_var.get()
        for var in self.scenario_vars.values():
            var.set(state)

    def toggle_all_data_types(self):
        state = self.all_types_var.get()
        for var in self.data_type_vars.values():
            var.set(state)

    def validate_selections(self):
        # Validate scenarios
        selected_scenarios = [id for id, var in self.scenario_vars.items() if var.get()]
        if not selected_scenarios:
            messagebox.showerror("Error", "Please select at least one scenario.")
            return False
        
        # Validate data types
        selected_types = [key for key, var in self.data_type_vars.items() if var.get()]
        if not selected_types:
            messagebox.showerror("Error", "Please select at least one data type.")
            return False
        
        # Validate number of instances
        try:
            num_instances = int(self.instances_var.get())
            if not (1 <= num_instances <= 10000):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Number of instances must be between 1 and 10000.")
            return False
        
        return True

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.root.update_idletasks()

    def start_generation(self):
        if not self.validate_selections():
            return
        
        self.generate_btn.state(['disabled'])
        self.status_var.set("Generating data...")
        
        # Get selected scenarios
        selected_scenarios = [
            scenario for scenario in gsd.SCENARIOS
            if self.scenario_vars[scenario['id']].get()
        ]
        
        # Get selected data types
        selected_types = {
            key: var.get()
            for key, var in self.data_type_vars.items()
        }
        
        # Get number of instances
        num_instances = int(self.instances_var.get())
        
        # Start generation in a separate thread
        thread = threading.Thread(
            target=self.generate_data,
            args=(selected_scenarios, selected_types, num_instances)
        )
        thread.start()

    def generate_data(self, selected_scenarios, selected_types, num_instances):
        try:
            # Create directories
            dirs = []
            if selected_types['text']: dirs.append('text_data')
            if selected_types['image']: dirs.append('image_data')
            if selected_types['audio']: dirs.append('audio_data')
            if selected_types['video']: dirs.append('video_data')
            if selected_types['physiological']: dirs.append('physiological_data')
            if selected_types['database']: dirs.append('metadata')
            
            for dir_name in dirs:
                os.makedirs(dir_name, exist_ok=True)
            
            # Create database if needed
            if selected_types['database']:
                gsd.create_database()
            
            # Prepare tasks
            tasks = []
            sample_id_counter = 0
            
            for scenario in selected_scenarios:
                for _ in range(num_instances):
                    sample_id_counter += 1
                    scenario_data = {
                        'SampleID': sample_id_counter,
                        'ScenarioID': scenario['id'],
                        'ScenarioName': scenario['name'],
                        'ScenarioDescription': scenario['description'],
                        'Emergency': scenario['emergency']
                    }
                    tasks.append((scenario_data, selected_types))
            
            # Generate data
            total_tasks = len(tasks)
            for i, task in enumerate(tasks):
                gsd.generate_sample(*task)
                self.root.after(0, self.update_progress, i + 1, total_tasks)
            
            self.status_var.set("Data generation completed successfully!")
            messagebox.showinfo("Success", 
                              f"Generated {total_tasks} samples for {len(selected_scenarios)} scenarios.")
            
        except Exception as e:
            self.status_var.set("Error during generation!")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        finally:
            self.generate_btn.state(['!disabled'])

def main():
    root = tk.Tk()
    app = SyntheticDataGeneratorGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main() 