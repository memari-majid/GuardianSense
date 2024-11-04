import streamlit as st
import generate_synthetic_data as gsd
import os

def create_streamlit_gui():
    st.set_page_config(page_title="Synthetic Medical Data Generator", layout="wide")
    
    st.title("Synthetic Medical Data Generator")
    
    # Create two columns for scenarios and data types
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Select Scenarios")
        
        # Select All Scenarios checkbox
        select_all_scenarios = st.checkbox("Select All Scenarios")
        
        # Individual scenario checkboxes
        selected_scenarios = []
        for scenario in gsd.SCENARIOS:
            if select_all_scenarios:
                is_selected = st.checkbox(
                    f"{scenario['name']} (Emergency: {scenario['emergency']})",
                    value=True,
                    key=f"scenario_{scenario['id']}"
                )
            else:
                is_selected = st.checkbox(
                    f"{scenario['name']} (Emergency: {scenario['emergency']})",
                    key=f"scenario_{scenario['id']}"
                )
            
            if is_selected:
                selected_scenarios.append(scenario)
    
    with col2:
        st.header("Select Data Types")
        
        # Select All Data Types checkbox
        select_all_types = st.checkbox("Select All Data Types")
        
        # Data type checkboxes
        data_types = {
            'text': 'Text data (descriptions and transcripts)',
            'image': 'Image data (facial expressions)',
            'audio': 'Audio data (synthesized speech)',
            'video': 'Video data (simple animations)',
            'physiological': 'Physiological data (vital signs)',
            'database': 'Database entries (metadata)'
        }
        
        selected_types = {}
        for key, description in data_types.items():
            if select_all_types:
                is_selected = st.checkbox(description, value=True, key=f"type_{key}")
            else:
                is_selected = st.checkbox(description, key=f"type_{key}")
            selected_types[key] = is_selected
    
    # Number of instances
    st.header("Number of Instances")
    num_instances = st.number_input(
        "Instances per scenario",
        min_value=1,
        max_value=10000,
        value=100,
        step=1
    )
    
    def validate_selections():
        if not selected_scenarios:
            st.error("Please select at least one scenario.")
            return False
        
        if not any(selected_types.values()):
            st.error("Please select at least one data type.")
            return False
        
        return True

    # Generate button
    if st.button("Generate Data"):
        if validate_selections():
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
                
                # Generate data with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_tasks = len(tasks)
                for i, task in enumerate(tasks):
                    gsd.generate_sample(*task)
                    # Update progress
                    progress = (i + 1) / total_tasks
                    progress_bar.progress(progress)
                    status_text.text(f"Generating data... {i + 1}/{total_tasks}")
                
                st.success(f"Generated {total_tasks} samples for {len(selected_scenarios)} scenarios!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_streamlit_gui() 