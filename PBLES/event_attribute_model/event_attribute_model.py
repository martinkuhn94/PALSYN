import pandas as pd


def build_attribute_model(df):
    """Build the attribute model from the DataFrame with event name prefixes."""

    # Create empty DataFrame with columns "Current State" and "Next State"
    base_model = pd.DataFrame(columns=['Current State', 'Next State'])

    # Drop 'case:concept:name' column
    df = df.drop(columns=['case:concept:name'])

    # Extract column names
    column_names = df.columns

    # Get all unique event names
    event_names = df['concept:name'].unique()

    for event_name in event_names:
        # Iterate over the columns for current and next state
        for i in range(len(column_names) - 1):
            current_state = f"{event_name}=={column_names[i]}"
            next_state = f"{event_name}=={column_names[i + 1]}"
            new_row = pd.DataFrame({'Current State': [current_state], 'Next State': [next_state]})
            base_model = pd.concat([base_model, new_row], ignore_index=True)

        # Add "START==START" for the beginning of each event block
        start_row = pd.DataFrame(
            {'Current State': ["START==START"], 'Next State': [f"{event_name}=={column_names[0]}"]})
        base_model = pd.concat([start_row, base_model], ignore_index=True)

        # Add "END==END" for the last row of each event block
        end_row = pd.DataFrame({'Current State': [f"{event_name}=={column_names[-1]}"], 'Next State': ["END==END"]})
        base_model = pd.concat([base_model, end_row], ignore_index=True)

    for event_name in event_names:
        for event_name_2 in event_names:
            end_row = pd.DataFrame({'Current State': [f"{event_name}=={column_names[-1]}"],
                                    'Next State': [f"{event_name_2}==concept:name"]})
            base_model = pd.concat([base_model, end_row], ignore_index=True)

    return base_model
