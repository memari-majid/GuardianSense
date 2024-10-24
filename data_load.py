# data_loading_and_querying.py

# -------------------------------
# Import Necessary Libraries
# -------------------------------

import sqlite3
import pandas as pd

# -------------------------------
# Database Interaction Functions
# -------------------------------

def connect_db():
    """
    Creates a connection to the SQLite database.

    Returns:
    - sqlite3.Connection: Database connection object.
    """
    conn = sqlite3.connect('metadata/dataset_metadata.db')
    return conn

def get_all_metadata():
    """
    Retrieves all metadata from the database.

    Returns:
    - pd.DataFrame: DataFrame containing all metadata records.
    """
    conn = connect_db()
    df = pd.read_sql_query('SELECT * FROM metadata', conn)
    conn.close()
    return df

def get_emergency_scenarios():
    """
    Retrieves all emergency scenarios from the database.

    Returns:
    - pd.DataFrame: DataFrame containing metadata for emergency scenarios.
    """
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM metadata WHERE Emergency='Yes'", conn)
    conn.close()
    return df

def get_scenario_by_name(scenario_name):
    """
    Retrieves metadata for a specific scenario by name.

    Parameters:
    - scenario_name (str): The name of the scenario to look up.

    Returns:
    - pd.DataFrame: DataFrame containing metadata for the specified scenario.
    """
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM metadata WHERE ScenarioName=?", conn, params=(scenario_name,))
    conn.close()
    return df

def get_samples_by_condition(condition):
    """
    Retrieves samples that meet a specific SQL condition.

    Parameters:
    - condition (str): SQL condition string (e.g., "SystolicBP > 140").

    Returns:
    - pd.DataFrame: DataFrame containing metadata for samples meeting the condition.
    """
    conn = connect_db()
    query = f"SELECT * FROM metadata WHERE {condition}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# -------------------------------
# Example Usage
# -------------------------------

if __name__ == '__main__':
    # Retrieve all metadata
    all_metadata = get_all_metadata()
    print("Total samples:", len(all_metadata))

    # Retrieve emergency scenarios
    emergency_scenarios = get_emergency_scenarios()
    print("Emergency scenarios:", len(emergency_scenarios))

    # Retrieve metadata for 'Heart Attack' scenarios
    heart_attack_samples = get_scenario_by_name('Heart Attack')
    print("Heart Attack samples:", len(heart_attack_samples))

    # Display some sample data
    print(heart_attack_samples.head())
