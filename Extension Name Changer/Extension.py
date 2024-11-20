import os

# Change to the directory containing the images
os.chdir('C:/Users/Admin/Desktop/AI/Day_13/images/Harry potter')

# Initialize a counter
i = 1

# Loop through all files in the directory
for file in os.listdir():
    src = file
    dst = "jumanji" + str(i) + ".png"
    
    # Rename the file
    os.rename(src, dst)
    
    # Increment the counter
    i += 1

printf("Successfully Converted")