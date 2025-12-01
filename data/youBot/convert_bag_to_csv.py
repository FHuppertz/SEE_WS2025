import rosbag
import numpy as np
import sys
import os
import glob

def get_poses(bagfile):
    """
    Opens a ROS Bag file, extracts X, Y, and Theta poses from the 
    /end_effector_pose topic, and returns them as a numpy array.
    """
    try:
        # Open the bag file
        bag = rosbag.Bag(bagfile)
        poses = []
        
        # Read messages from the specified topic
        for topic, msg, t in bag.read_messages(topics=['/end_effector_pose']):
            # Assuming the message object has x, y, and theta attributes
            poses.append([msg.x, msg.y, msg.theta])

        # Close the bag file
        bag.close()
        
    except Exception as e:
        # Handle exceptions (e.g., file not found, bag corrupted, topic missing)
        print('Error processing %s: %s' % (bagfile, e))
        # Ensure bag is closed if an error occurred before reaching the successful close
        try:
            bag.close()
        except:
            pass # Ignore if bag wasn't successfully opened
        return None

    # Convert the list of poses to a NumPy array
    poses = np.array(poses)
    print("Successfully extracted data from: %s" % bagfile)
    # print(poses) # Uncomment for detailed debug output
    return poses

def main():
    """
    Main function to find all bag files in a directory and save their 
    end effector poses to a 'csv' subdirectory.
    """
    # 1. Argument Check
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <root_directory>")
        print("specify root directory")
        return
        
    root_dir = str(sys.argv[1])
    
    # 2. Directory Setup
    csv_path = os.path.join(root_dir, 'csv')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        print("Created output directory: %s" % csv_path)
        
    # 3. File Discovery
    files = sorted(glob.glob(os.path.join(root_dir, '*.bag')))
    if not files:
        print("No .bag files found in %s" % root_dir)
        return

    print("Found %d bag files to process." % len(files))

    # 4. Processing Loop (Fixed Indentation)
    for f in files:
        # Generate the output filename
        bag_basename = os.path.basename(f)
        filename = bag_basename[:-4] + '.csv' # Replace .bag with .csv
        
        # Extract data
        data = get_poses(f)
        
        # Save data IF extraction was successful (THIS BLOCK WAS MOVED INSIDE THE LOOP)
        if data is not None:
            output_filepath = os.path.join(csv_path, filename)
            np.savetxt(output_filepath, data, delimiter=',')
            print("Saved data to: %s" % output_filepath)
        else:
            print("Skipping save for %s due to extraction error." % bag_basename)

if __name__ == '__main__':
    main()