def extract(direction, threshold, data_df):
    """
    Extracts end points from encoder data based on a change in 'X'
    and a condition on 'Y'.
    
    Args:
        direction (str): 'left', 'right', or 'straight'.
        threshold (float): The minimum change in 'X' to trigger an end point.
        data_df (pd.DataFrame): The DataFrame containing 'X' and 'Y' columns.
        
    Returns:
        pd.DataFrame: A new DataFrame containing only the end point rows.
    """
    
    last_x = 0
    end_points = [] # This will store the rows that are end points

    # Use .iterrows() to loop through a DataFrame row by row
    # 'row' will be a pandas Series containing all data for that row
    for index, row in data_df.iterrows():
        
        # Access data using column names
        current_x = row['X']
        current_y = row['Y']
        
        # Your logic was already correct!
        # We just needed to apply it to the right variables.
        
        if direction == 'left':
            if abs(current_x - last_x) > threshold and current_y < -10:
                end_points.append(row.to_dict()) # Add the full row's data
                last_x = current_x # Update last_x *only* when a point is found
            
        elif direction == 'right':
            if abs(current_x - last_x) > threshold and current_y > 10:
                end_points.append(row.to_dict())
                last_x = current_x # Update last_x *only* when a point is found
            
        elif direction == 'straight':
            if abs(current_x - last_x) > threshold and current_y > -12 and current_y < 5:
                end_points.append(row.to_dict())
                last_x = current_x # Update last_x *only* when a point is found
            
    # Convert the list of dictionaries back into a clean DataFrame
    if not end_points:
        # Return an empty DataFrame with the same columns if no points are found
        return pd.DataFrame(columns=data_df.columns)
        
    return pd.DataFrame(end_points)


left = extract('left', 31.5, encoder_df1)

print("left end points list below:")

print(left)
print(f"lenght of left list: {len(left)}")
print(f"type of left list: {type(left)}")

raise Exception("left end points extracted")

