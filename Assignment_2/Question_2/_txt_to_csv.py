# importing panda library
import pandas as pd

# readinag given csv file
# and creating dataframe
dataframe1 = pd.read_csv("Assignment_2\Question_2\deformation_over_time.txt", delim_whitespace=True)

# storing this dataframe in a csv file
dataframe1.to_csv('Assignment_2\Question_2\step0.csv',
				index = None,)
