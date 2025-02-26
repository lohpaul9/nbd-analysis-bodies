from dataclasses import dataclass
from matplotlib.pylab import gamma
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist

MALE_DATA = 'male_data.csv'
FEMALE_DATA = 'female_data.csv'

GENDER_OPTIONS = {
    'Male': 1,
    'Female': 2,
    'Both': 3
}

def age_filter(df: pd.DataFrame, age_option) -> pd.DataFrame:
    if isinstance(age_option, list):
        return df[df['RSCRAGE'].between(age_option[0], age_option[1], inclusive="both")]
    else:
        return df[df['RSCRAGE'] == age_option]


RACE_OPTIONS = {
    "Other": 1,
    "White": 2,
    "Black": 3,
    "Hispanic": 4
}

def RACE_FILTER(df: pd.DataFrame, race: int) -> pd.DataFrame:
    return df[df['RSCRRACE'] == race]

RELIGION_OPTIONS = {
    "No Religion": 1,
    "Catholic": 2,
    "Protestant": 3,
    "Other Religion": 4
}

def RELIGION_FILTER(df: pd.DataFrame, religion: int = RELIGION_OPTIONS['No Religion']) -> pd.DataFrame:
    return df[df['RELIGION'] == religion]

MARITAL_STATUS_OPTIONS = {
    "Married": 1,
    "Separated / Divorced": 2,
    "Never Married": 3
}

def MARITAL_STATUS_FILTER(df: pd.DataFrame, marital_status: int = MARITAL_STATUS_OPTIONS['Married']) -> pd.DataFrame:
    if marital_status == MARITAL_STATUS_OPTIONS['Married']:
        return df[df['FMARIT'] == 1]
    elif marital_status == MARITAL_STATUS_OPTIONS['Separated / Divorced']:
        return df[(df['FMARIT'] == 3) | (df['FMARIT'] == 4)]
    else:
        return df[df['FMARIT'] == 5]

SEPARATED_MARITAL_STATUS_OPTIONS = {
    "Divorced": 1,
    "Separated": 2,
    "Widowed": 3
}

def SEPARATE_MARITAL_STATUS_FILTER(df: pd.DataFrame, separate: int = SEPARATED_MARITAL_STATUS_OPTIONS['Divorced']) -> pd.DataFrame:
    if separate == SEPARATED_MARITAL_STATUS_OPTIONS['Divorced']:
        return df[df['MAREND01'] == 1]
    elif separate == SEPARATED_MARITAL_STATUS_OPTIONS['Separated']:
        return df[df['MAREND01'] == 2]
    else:
        return df[df['MAREND01'] == 3]
    
EDUCATION_OPTIONS = {
    "No High School Diploma": 2,
    "High School Graduation At Least": 3,
    "College Degree At Least": 6,
    "Masters Degree at Least": 8
}

def EDUCATION_FILTER(df: pd.DataFrame, education: int) -> pd.DataFrame:
    if education == EDUCATION_OPTIONS['No High School Diploma']:
        return df[df['HIEDUC'] <= education]
    elif education == EDUCATION_OPTIONS['High School Graduation At Least']:
        return df[df['HIEDUC'] >= education]
    elif education == EDUCATION_OPTIONS['College Degree At Least']:
        return df[df['HIEDUC'] >= education]
    elif education == EDUCATION_OPTIONS['Masters Degree at Least']:
        return df[df['HIEDUC'] >= education]

INTACT_FAMILY_18_OPTIONS = {
    "Intact": 1,
    "Non-Intact": 2
}

def INTACT_FAMILY_18_FILTER(df: pd.DataFrame, intact: int = INTACT_FAMILY_18_OPTIONS['Intact']) -> pd.DataFrame:
    if intact == INTACT_FAMILY_18_OPTIONS['Intact']:
        return df[df['INTACT18'] == 1]
    elif intact == INTACT_FAMILY_18_OPTIONS['Non-Intact']:
        return df[df['INTACT18'] == 2]
    

NO_MOTHER_FIGURE_OPTIONS = {
    "Yes": 1,
    "No": 2
}

def NO_MOTHER_FIGURE_FILTER(df: pd.DataFrame, no_mother: int = NO_MOTHER_FIGURE_OPTIONS['Yes']) -> pd.DataFrame:
    if no_mother == NO_MOTHER_FIGURE_OPTIONS['Yes']:
        return df[(df['LVSIT14F'] != 3)]
    elif no_mother == NO_MOTHER_FIGURE_OPTIONS['No']:
        return df[df['LVSIT14F'] == 3]
    
NO_FATHER_FIGURE_OPTIONS = {
    "Yes": 1,
    "No": 2
}

def NO_FATHER_FIGURE_FILTER(df: pd.DataFrame, no_father: int = NO_FATHER_FIGURE_OPTIONS['Yes']) -> pd.DataFrame:
    if no_father == NO_FATHER_FIGURE_OPTIONS['Yes']:
        return df[(df['LVSIT14M'] != 3)]
    elif no_father == NO_FATHER_FIGURE_OPTIONS['No']:
        return df[df['LVSIT14M'] == 3]
    

PARENTS_MARRIED_AT_BIRTH_OPTIONS = {
    "Yes": 1,
    "No": 5
}

def PARENTS_MARRIED_AT_BIRTH_FILTER(df: pd.DataFrame, parents_married: int = PARENTS_MARRIED_AT_BIRTH_OPTIONS['Yes']) -> pd.DataFrame:
    if parents_married == PARENTS_MARRIED_AT_BIRTH_OPTIONS['Yes']:
        return df[df['PARMARR'] == 1]
    elif parents_married == PARENTS_MARRIED_AT_BIRTH_OPTIONS['No']:
        return df[df['PARMARR'] == 5]
    
SUSPENDED_FROM_SCHOOL_OPTIONS_U_25 = {
    "Yes": 1,
    "No": 5
}

def SUSPENDED_FROM_SCHOOL_FILTER(df: pd.DataFrame, suspended: int = SUSPENDED_FROM_SCHOOL_OPTIONS_U_25['Yes']) -> pd.DataFrame:
    if suspended == SUSPENDED_FROM_SCHOOL_OPTIONS_U_25['Yes']:
        return df[df['EVSUSPEN'] == 1]
    elif suspended == SUSPENDED_FROM_SCHOOL_OPTIONS_U_25['No']:
        return df[df['EVSUSPEN'] == 5]

LIVES_WITH_ANY_PARENTS_CURRENTLY_OPTIONS = {
    "Yes": 1,
    "No": 2
}

def LIVES_WITH_ANY_PARENTS_CURRENTLY_FILTER(df: pd.DataFrame, lives_with_parents: int = LIVES_WITH_ANY_PARENTS_CURRENTLY_OPTIONS['Yes']) -> pd.DataFrame:
    if lives_with_parents == LIVES_WITH_ANY_PARENTS_CURRENTLY_OPTIONS['Yes']:
        return df[df['WTHPARNW'] == 1]
    elif lives_with_parents == LIVES_WITH_ANY_PARENTS_CURRENTLY_OPTIONS['No']:
        return df[df['WTHPARNW'] == 2]
    
LIVE_AWAY_FRON_PARENTS_BEFORE_18_OPTIONS = {
    "Yes": 1,
    "No": 5
}

def LIVED_AWAY_FROM_PARENTS_BEFORE_18_FILTER(df: pd.DataFrame, live_away: int = LIVE_AWAY_FRON_PARENTS_BEFORE_18_OPTIONS['Yes']) -> pd.DataFrame:
    if live_away == LIVE_AWAY_FRON_PARENTS_BEFORE_18_OPTIONS['Yes']:
        return df[df['ONOWN'] == 1]
    elif live_away == LIVE_AWAY_FRON_PARENTS_BEFORE_18_OPTIONS['No']:
        return df[df['ONOWN'] == 5]

EMOTABUSE_BY_PARENTS_OPTIONS = {
    "Yes": 1,
    "No": 5
}

def EMOTABUSE_FILTER(df: pd.DataFrame, emot_abuse: int = EMOTABUSE_BY_PARENTS_OPTIONS['Yes']) -> pd.DataFrame:
    if emot_abuse == EMOTABUSE_BY_PARENTS_OPTIONS['Yes']:
        return df[(df['EMOTABUSE'] >= 2) & (df['EMOTABUSE'] <= 5)]
    elif emot_abuse == EMOTABUSE_BY_PARENTS_OPTIONS['No']:
        return df[df['EMOTABUSE'] == 1]

FOSTER_CARE_OPTIONS = {
    "Yes": 1,
    "No": 5
}

def FOSTER_CARE_FILTER(df: pd.DataFrame, foster_care: int = FOSTER_CARE_OPTIONS['Yes']) -> pd.DataFrame:
    if foster_care == FOSTER_CARE_OPTIONS['Yes']:
        return df[df['FOSTEREV'] == 1]
    elif foster_care == FOSTER_CARE_OPTIONS['No']:
        return df[df['FOSTEREV'] == 5]


SEXABUSE_HISTORY_OPTIONS = {
    "Yes": 1,
    "No": 5
}

def SEXABUSE_HISTORY_FILTER(df: pd.DataFrame, sex_abuse: int) -> pd.DataFrame:
    if sex_abuse == SEXABUSE_HISTORY_OPTIONS['Yes']:
        return df[(df['SEXABUSE'] >= 2) & (df['SEXABUSE'] <= 5)]
    elif sex_abuse == SEXABUSE_HISTORY_OPTIONS['No']:
        return df[df['SEXABUSE'] == 1]
    

PHYSABUSE_BY_PARENTS_OPTIONS = {
    "Yes": 1,
    "No": 5
}

def PHYSABUSE_FILTER(df: pd.DataFrame, phys_abuse: int = PHYSABUSE_BY_PARENTS_OPTIONS['Yes']) -> pd.DataFrame:
    if phys_abuse == PHYSABUSE_BY_PARENTS_OPTIONS['Yes']:
        return df[(df['PHYSABUSE'] >= 2) & (df['PHYSABUSE'] <= 5)]
    elif phys_abuse == PHYSABUSE_BY_PARENTS_OPTIONS['No']:
        return df[df['PHYSABUSE'] == 1]



GENDER_LOOKUP_KEY = "Gender" 
AGE_LOOKUP_KEY = "Age"
RACE_LOOKUP_KEY = "Race"
RELIGION_LOOKUP_KEY = "Religion"
MARITAL_STATUS_LOOKUP_KEY = "Current Marital Status"
SEPARATED_MARITAL_STATUS_LOOKUP_KEY = "Reason for 1st Marriage Ending"
EDUCATION_LOOKUP_KEY = "Education Level"
INTACT_FAMILY_18_LOOKUP_KEY = "Intact Family by 18"
NO_MOTHER_FIGURE_LOOKUP_KEY = "Had a Mother Figure"
NO_FATHER_FIGURE_LOOKUP_KEY = "Had a Father Figure"
PARENTS_MARRIED_AT_BIRTH_LOOKUP_KEY = "Parents were Married when born"
SUSPENDED_FROM_SCHOOL_U_25_LOOKUP_KEY = "Suspended from School (age <= 25)"
LIVES_WITH_ANY_PARENTS_CURRENTLY_LOOKUP_KEY = "Lives with Parents Currently"
LIVED_AWAY_FROM_PARENTS_BEFORE_18_LOOKUP_KEY = "Lived away from Parents before 18"
EMOTABUSE_BY_PARENTS_LOOKUP_KEY = "Ever Emotionally Abused by Parents"
SEXABUSE_HISTORY_LOOKUP_KEY = "Ever Sexually Abused"
PHYSABUSE_BY_PARENTS_LOOKUP_KEY = "Ever Physically Abused by Parents"
FOSTER_CARE_LOOKUP_KEY = "Has been in Foster Care"

DEFAULT_FILTER_OPTIONS = {
    (GENDER_LOOKUP_KEY, GENDER_OPTIONS["Both"]),
    (AGE_LOOKUP_KEY, (25, 30)),
}

FILTER_LOOKUP_MAP = {
    GENDER_LOOKUP_KEY: (GENDER_OPTIONS, None),
    AGE_LOOKUP_KEY: (None, age_filter),
    RACE_LOOKUP_KEY: (RACE_OPTIONS, RACE_FILTER),
    RELIGION_LOOKUP_KEY: (RELIGION_OPTIONS, RELIGION_FILTER),
    MARITAL_STATUS_LOOKUP_KEY: (MARITAL_STATUS_OPTIONS, MARITAL_STATUS_FILTER),
    SEPARATED_MARITAL_STATUS_LOOKUP_KEY: (SEPARATED_MARITAL_STATUS_OPTIONS, SEPARATE_MARITAL_STATUS_FILTER),
    EDUCATION_LOOKUP_KEY: (EDUCATION_OPTIONS, EDUCATION_FILTER),
    INTACT_FAMILY_18_LOOKUP_KEY: (INTACT_FAMILY_18_OPTIONS, INTACT_FAMILY_18_FILTER),
    NO_MOTHER_FIGURE_LOOKUP_KEY: (NO_MOTHER_FIGURE_OPTIONS, NO_MOTHER_FIGURE_FILTER),
    NO_FATHER_FIGURE_LOOKUP_KEY: (NO_FATHER_FIGURE_OPTIONS, NO_FATHER_FIGURE_FILTER),
    PARENTS_MARRIED_AT_BIRTH_LOOKUP_KEY: (PARENTS_MARRIED_AT_BIRTH_OPTIONS, PARENTS_MARRIED_AT_BIRTH_FILTER),
    SUSPENDED_FROM_SCHOOL_U_25_LOOKUP_KEY: (SUSPENDED_FROM_SCHOOL_OPTIONS_U_25, SUSPENDED_FROM_SCHOOL_FILTER),
    LIVES_WITH_ANY_PARENTS_CURRENTLY_LOOKUP_KEY: (LIVES_WITH_ANY_PARENTS_CURRENTLY_OPTIONS, LIVES_WITH_ANY_PARENTS_CURRENTLY_FILTER),
    LIVED_AWAY_FROM_PARENTS_BEFORE_18_LOOKUP_KEY: (LIVE_AWAY_FRON_PARENTS_BEFORE_18_OPTIONS, LIVED_AWAY_FROM_PARENTS_BEFORE_18_FILTER),
    EMOTABUSE_BY_PARENTS_LOOKUP_KEY: (EMOTABUSE_BY_PARENTS_OPTIONS, EMOTABUSE_FILTER),
    SEXABUSE_HISTORY_LOOKUP_KEY: (SEXABUSE_HISTORY_OPTIONS, SEXABUSE_HISTORY_FILTER),
    PHYSABUSE_BY_PARENTS_LOOKUP_KEY: (PHYSABUSE_BY_PARENTS_OPTIONS, PHYSABUSE_FILTER),
    FOSTER_CARE_LOOKUP_KEY: (FOSTER_CARE_OPTIONS, FOSTER_CARE_FILTER),
}

df1 = pd.read_csv(MALE_DATA, low_memory=True)
df2 = pd.read_csv(FEMALE_DATA, low_memory=True)

def read_data(filters):
    if GENDER_LOOKUP_KEY not in filters:
        filters[GENDER_LOOKUP_KEY] = GENDER_OPTIONS['Both']

    gender = filters[GENDER_LOOKUP_KEY]
    filters.pop(GENDER_LOOKUP_KEY)
    
    # Combine the dataframes if needed
    if gender == GENDER_OPTIONS['Both']:
        df_combined = pd.concat([df1, df2], ignore_index=True)
    else:
        df_combined = df1 if gender == GENDER_OPTIONS['Male'] else df2

    # Transformation 1: Set OPPLIFENUM = 0 where OPPSEXANY == 5
    df_combined.loc[df_combined['OPPSEXANY'] == 5, 'OPPLIFENUM'] = 0

    # Transformation 2: Filter by age (RSCRAGE between 25 and 30) and ORIENT == 2
    for filter_key, filter_value in filters.items():
        df_combined = FILTER_LOOKUP_MAP[filter_key][1](df_combined, filter_value)

    df_filtered = df_combined[df_combined['OPPLIFENUM'].between(0, 50)]

    # Transformation 4: Aggregate counts for OPPLIFENUM values.
    # First, count the occurrences of each value
    opplifenum_counts = df_filtered['OPPLIFENUM'].value_counts().sort_index()

    # Ensure the full range 0-50 is present (fill missing with 0)
    full_range = pd.Series(index=range(0, 51), data=0)
    opplifenum_counts_filled = full_range.add(opplifenum_counts, fill_value=0).astype(int)
    opplifenum_counts_filled = opplifenum_counts_filled.sort_index()

    # Convert the series to a DataFrame for clarity
    cleaned_data_df = opplifenum_counts_filled.reset_index()
    cleaned_data_df.columns = ['OPPLIFENUM', 'Count']

    return cleaned_data_df


def test_all_filter_combinations():
    # Skip age since it's a special case with ranges
    filter_keys = [key for key in FILTER_LOOKUP_MAP.keys() if key != AGE_LOOKUP_KEY]
    
    results = []
    
    # Test each filter individually first
    for key in filter_keys:
        options_dict = FILTER_LOOKUP_MAP[key][0]
        if options_dict is None:
            continue
            
        for option_name, option_value in options_dict.items():
            try:
                filter_options = {
                    'GENDER': GENDER_OPTIONS['both'],  # Default to both genders
                    key: option_value
                }
                
                df = read_data(filter_options)
                count = df['Count'].sum()
                
                results.append({
                    'filters': f"{key}={option_name}",
                    'count': count,
                    'status': 'empty' if count == 0 else 'ok'
                })
            except Exception as e:
                raise e
                results.append({
                    'filters': f"{key}={option_name}",
                    'count': 0,
                    'status': f'error: {str(e)}'
                })

    # Print results in a table format
    print("\nFilter Test Results:")
    headers = ['Filters', 'Count', 'Status']
    table_data = [[r['filters'], r['count'], r['status']] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Print summary
    total_tests = len(results)
    errors = sum(1 for r in results if r['status'].startswith('error'))
    empty = sum(1 for r in results if r['status'] == 'empty')
    
    print(f"\nSummary:")
    print(f"Total tests: {total_tests}")
    print(f"Errors: {errors}")
    print(f"Empty results: {empty}")
    print(f"Successful filters: {total_tests - errors - empty}")

if __name__ == "__main__":
    test_all_filter_combinations()






"""
Interesting demographics to filter for:
OppSexAny: OPPSEXANY
OppLifeNum: OPPLIFENUM

(Age range and gender range)
Age: RSCRAGE
Gender: by filename
Race: RSCRRACE
Orientation: ORIENT
Religion: RELIGION
Income: TOTINCR

(Marital status and family)
Current Marital status: FMARIT
Has been separated / divorced before: MAREND01
Ever married: EVRMARRY

(Education)
Has a bachelor's: HIEDUC

(Family)
Intact family before 18: INTACT18
No mother figure: LVSIT14F
No father figure: LVSIT14M
Ever been suspended from school (<25 age): EVSUSPEN
Lives with parents currently: WTHPARNW
Lived away from parents before 18: ONOWN
Abuse: SEXABUSE, EMOTABUSE, PHYSABUSE
Foster care: FOSTEREV

Interesting questions to ask:
How significant do family effects play in the life of individual? 
- Abuse? Multiple kinds of abuse?
- Intact family?
- Living away from parents?

"""