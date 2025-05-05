# load pandas dataframe from csv file
import pandas as pd

# load data
BASE_DIR = '/nfs/home/zamilp/SKO'
df = pd.read_csv(f'{BASE_DIR}/queryv3_abstract_unpaywall.csv')

df.info()


# split 'predicateLabel' by ','
df['predicateLabel'] = df['predicateLabel'].str.split(',')

# remove leading and trailing whitespaces from 'predicateLabel'
df['predicateLabel'] = df['predicateLabel'].apply(lambda x: [i.strip() for i in x])

# Rename columns for consistency
df = df.rename(columns={'unpaywall_abstract': 'abstract', 'paper_title': 'title', 'predicateLabel': 'predicate_label'})

# remove duplicates in 'predicateLabel'
df['predicate_label'] = df['predicate_label'].apply(lambda x: list(set(x)))

# rows with 'predicateLabel' length more than 3
df = df[df['predicate_label'].str.len() > 3]

# remove 'Unnamed: 0.1' column
df.drop(columns=['Unnamed: 0.1'], inplace=True)

# remove leading and trailing whitespaces from 'research_field_label'
df['research_field_label'] = df['research_field_label'].str.strip()


# remove rows with 'research_field_label' is 'Science'
df = df[df['research_field_label'] != 'Science']

taxonomy_df = pd.read_csv(f'{BASE_DIR}/Taxonomy_and_Fields.csv')

# Extract unique fields and taxonomy information from the reference data
taxonomy_fields_mapping = taxonomy_df[['Field', 'Taxonomy']].drop_duplicates()

# Define a function to determine the correct 'Field' and 'Taxonomy' based on the 'research_field_label' in SKO data
def get_correct_field_and_taxonomy(label):
    # Check if the label matches directly with the 'Field'
    match_field = taxonomy_fields_mapping[taxonomy_fields_mapping['Field'].str.lower() == label.lower()]
    if not match_field.empty:
        return match_field.iloc[0]['Field'], match_field.iloc[0]['Taxonomy']
    
    # Check if the label matches with any of the 'sub-field'
    match_subfield = taxonomy_df[taxonomy_df['sub-field'].str.contains(label, case=False, na=False)]
    if not match_subfield.empty:
        return match_subfield.iloc[0]['Field'], match_subfield.iloc[0]['Taxonomy']
    
    return None, None

# Apply the function to the SKO dataframe to create new columns 'Field' and 'Taxonomy'
df[['Field', 'Taxonomy']] = df['research_field_label'].apply(lambda label: pd.Series(get_correct_field_and_taxonomy(label)))

# remove rows where 'Field' and 'Taxonomy' is None
df = df.dropna(subset=['Field', 'Taxonomy']).reset_index(drop=True)


# Find the unique 'Field' values from both dataframes
fields_in_df = df['Field'].dropna().unique()
fields_in_taxonomy_df = taxonomy_df['Field'].unique()
print(f"Total unique fields in SKO data: {len(fields_in_df)}")
print(f"Total unique fields in Taxonomy data: {len(fields_in_taxonomy_df)}")

# Determine the fields in taxonomy_df that are missing in df
missing_fields = set(fields_in_taxonomy_df) - set(fields_in_df)

# Count the number of missing fields
missing_fields_count = len(missing_fields)
print(f"Total missing fields in SKO data: {missing_fields_count}")


# save df to pickle
df.to_pickle(f'{BASE_DIR}/SKO_with_taxonomy.pkl')