from matplotlib.pylab import gamma
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from data_reader import *
from scipy.stats import gamma
import numpy as np
from json_tricks import dumps, loads
from scipy.stats import chi2



def compute_nbd_probs(alpha, r, max_val=49, t=1.0):
    """
    Computes the negative binomial probabilities P_NB(x) for x=0,...,max_val using
    the recursive relationship.
    """
    probs = np.zeros(max_val + 1)
    probs[0] = (alpha / (alpha + t)) ** r
    for x in range(1, max_val + 1):
        # Using the ratio formula:
        ratio = (r + x - 1) / x * (t / (alpha + t))
        probs[x] = probs[x - 1] * ratio
    return probs

def compute_model_probs(theta, alpha, r, debug=False):
    """
    Computes the overall probabilities P(x) for x=0,...,50 for the spiked negative binomial model.
    For x = 0: P(0) = P_NB(0)
    For x = 1: P(1) = theta + (1-theta)*P_NB(1)
    For other x (2 <= x <= 49): P(x) = (1-theta)*P_NB(x)
    For x = 50: P(50) = 1 - sum(P(x) for x in 0..49)
    """
    # Compute the NB probabilities for x=0,...,49
    nb_probs = compute_nbd_probs(alpha, r, max_val=49, t=1.0)
    # check that this is close to 1
    if not np.isclose(nb_probs.sum(), 1.0, atol=1e-2):
        if debug:
            print(f"Sum of NB probabilities is not close to 1: {nb_probs.sum()} for alpha: {alpha}, r: {r}")
        if nb_probs.sum() > 1.0:
            raise ValueError("NB probabilities sum to greater than 1")
    
    # Initialize the model probabilities for x=0,...,50
    model_probs = np.zeros(51)
    
    # For x = 1, add the spike probability
    for x in range(0, 50):
        if x == 1:
            model_probs[1] = theta + (1 - theta) * nb_probs[1]
        else:
            model_probs[x] = (1 - theta) * nb_probs[x]

    # check that model_probs[:50].sum() is close to 1
    if not np.isclose(model_probs[:50].sum(), 1.0, atol=1e-2):
        # print(f"Sum of spiked model probabilities is not close to 1 before >=50: {model_probs[:50].sum()} for alpha: {alpha}, r: {r} theta: {theta}")
        if model_probs[:50].sum() > 1.0:
            raise ValueError("Model probabilities sum to greater than 1")
    
    # For x=50: bucket probability for all counts >= 50
    model_probs[50] = 1 - model_probs[:50].sum()

   
    return model_probs

def negative_log_likelihood(params, observed_counts):
    """
    Computes the negative log likelihood for the spiked negative binomial model.
    params: [theta, alpha, r]
      - theta must be in (0,1)
      - alpha > 0, r > 0
    observed_counts: array-like of counts for x=0,...,50.
    """
    theta, alpha, r = params
    # Compute the model probabilities for x=0,...,50
    probs = compute_model_probs(theta, alpha, r)
    
    # To avoid log(0), we add a very small constant
    eps = 1e-12
    log_probs = np.log(probs + eps)
    
    # Compute the negative log likelihood (note: we want to maximize log likelihood)
    nll = -np.sum(observed_counts * log_probs)
    return nll


def get_params(cleaned_data_df):
    # Extract observed counts as a numpy array
    observed_counts = cleaned_data_df['Count'].values

    # Define initial parameter guesses
    initial_params = [0.5, 1.0, 1.0]  # [theta, alpha, r]

    # Set parameter bounds: theta in (0,1), alpha > 0, r > 0
    bounds = [(1e-5, 1 - 1e-5), (1e-5, None), (1e-5, None)]

    # Run the minimization
    result = minimize(
        negative_log_likelihood,
        x0=initial_params,
        args=(observed_counts,),
        method='L-BFGS-B',
        bounds=bounds
    )

    # Extract the best-fit parameters
    theta_hat, alpha_hat, r_hat = result.x
    # print("Optimized Parameters:")
    # print(f"theta = {theta_hat}")
    # print(f"alpha = {alpha_hat}")
    # print(f"r = {r_hat}")
    return theta_hat, alpha_hat, r_hat

def calculate_chi_square_test(predicted_counts, actual_counts):
    """
    Calculate the chi-square goodness of fit test p-value
    
    Args:
        predicted_counts: array of predicted counts (not probabilities)
        actual_counts: array of actual counts
    
    Returns:
        tuple of (chi_square_stat, p_value)
    """
    # Calculate chi-square statistic
    chi_square_stat = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_counts, predicted_counts)
    )

    chi_square_stat = chi_square_stat / actual_counts.sum()
    
    # Degrees of freedom = number of cells (51) - number of parameters (3) - 1
    df = 51 - 3 - 1  # = 47
    
    # Calculate p-value using survival function (equivalent to 1 - cdf)
    # This is equivalent to CHISQ.DIST.RT in Excel
    p_value = chi2.sf(chi_square_stat, df)
    
    return chi_square_stat, p_value

def get_predictions(params, cleaned_data_df):
    theta_hat, alpha_hat, r_hat = params
    probs = compute_model_probs(theta_hat, alpha_hat, r_hat)
    
    # get the total number of people in the dataset
    total_people = cleaned_data_df['Count'].sum()

    # Calculate predicted counts
    predicted_counts = total_people * probs
    
    # Calculate chi-square test
    chi_square_stat, p_value = calculate_chi_square_test(
        predicted_counts,
        cleaned_data_df['Count'].values
    )

    # Add to dataframe
    cleaned_data_df['Predicted'] = predicted_counts
    cleaned_data_df['Predicted of 1000'] = 1000 * probs
    cleaned_data_df['Probs'] = probs
    cleaned_data_df['Cumulative Probs'] = np.cumsum(probs)
    cleaned_data_df['Data Probs'] = np.array(cleaned_data_df['Count']) / total_people
    
    return cleaned_data_df, chi_square_stat, p_value

def display_table_nicely(cleaned_data_df):
    # display the table in a nice format
    print(tabulate(cleaned_data_df, headers='keys', tablefmt='grid'))


def get_aggregate_stats(cleaned_data_df, params, chi_square_results=None):
    theta_hat, alpha_hat, r_hat = params
    probs = compute_model_probs(theta_hat, alpha_hat, r_hat)
    # E(X) = (1-theta) * rt / alpha
    mean = (1 - theta_hat) * r_hat * 1 / alpha_hat
    # get median using probs by checking when the cumulative sum of probs is >= 0.5
    cumulative_probs = np.cumsum(probs)
    # now get the FIRST index where the cumulative sum is >= 0.5
    median = np.where(cumulative_probs >= 0.5)[0][0]

    # get mode using probs by checking when the probability is the highest
    mode = np.argmax(probs)
    # get variance using probs
    variance = np.sum((np.arange(0, 51) - mean) ** 2 * probs)
    
    # Add chi-square results if available
    stats = {
        'mean': float(mean), 
        'median': float(median), 
        'mode': float(mode), 
        'variance': float(variance)
    }
    
    if chi_square_results:
        chi_square_stat, p_value = chi_square_results
        stats.update({
            'chi_square_stat': float(chi_square_stat),
            'chi_square_p_value': float(p_value)
        })
    
    return stats


def get_mixing_parameters_points(params):
    _, alpha_hat, r_hat = params
    # alpha_hat and r_hat are the parameters of an underlying mixing gamma distribution
    # we want to plot the pdf of this distribution
    x = np.linspace(0.01, 30, 300)
    # Use scipy.stats.gamma instead of matplotlib.pylab.gamma

    pdf = gamma_dist.pdf(x, a=r_hat, scale=1/alpha_hat)
    return x, pdf

def plot_mixing_parameters_points(experiments):
    for experiment in experiments:
        params = experiment['params']
        x, pdf = get_mixing_parameters_points(params)
        plt.plot(x, pdf, label=experiment['name'])
    plt.legend()
    plt.show()


def compute_lorenz_curve_points(params):
    """
    Computes the Lorenz curve points for a given set of p-values (percentiles) using
    the Negative Binomial-derived Gamma mixing distribution.

    Parameters:
    - params: Tuple of estimated parameters (theta, alpha_hat, r_hat).
    - p_values: Array-like, percentiles at which to evaluate the Lorenz curve.

    Returns:
    - lorenz_points: Corresponding Lorenz curve values for each p in p_values.
    """
    _, alpha_hat, r_hat = params  # Extract relevant parameters
    p_values = np.linspace(0, 1, 500)

    # Compute the inverse gamma CDF (quantile function) for each p
    gamma_inv = gamma.ppf(p_values, a=r_hat, scale=1/alpha_hat)

    # Compute the gamma CDF of the computed values with r+1 degrees of freedom
    lorenz_points = gamma.cdf(gamma_inv, a=r_hat + 1, scale=1/alpha_hat)

    return p_values, lorenz_points


def graph_predictions_vs_actual_for_multiple_completed_experiments(experiments):
    # note that 50 should be labelled as 50+ instead of 50
    # i think we should just use a string for the x axis labels
    x_axis = np.arange(0, 51)
    x_axis_labels = [str(x) for x in x_axis]
    x_axis_labels[50] = '50+'
    x_axis_title = 'Lifetime Number of Sexual Partners'

    # plot the actual counts
    # if len(experiments) == 1:
        # plt.plot(x_axis, experiments[0]['predictions']['Count'], label='Actual')
    
    for experiment in experiments:
        data = experiment['predictions']
        experiment_name = experiment['name']
        
        # now we just plot the predictions vs the actual for everything
        plt.plot(x_axis, data['Probs'], label=experiment_name + f" {data['Count'].sum()}")

    plt.xlabel(x_axis_title)
    plt.ylabel('Counts')
    plt.title(f'Comparisons between {", ".join([x["name"] for x in experiments])}')

    # also show the parameters for each experiment
    for experiment in experiments:
        params = experiment['params']
        # show below the plot somewhere
        plt.text(0.5, 0.5, f'{experiment_name}: theta = {params[0]}, alpha = {params[1]}, r = {params[2]}', ha='center', va='center')


    plt.legend()
    plt.show()

def graph_lorenz_curve_for_multiple_completed_experiments(experiments):
    for experiment in experiments:
        params = experiment['params']
        lorenz_points = compute_lorenz_curve_points(params)
        plt.plot(lorenz_points[0], lorenz_points[1], label=experiment['name'])

    plt.xlabel('Percentile')
    plt.ylabel('Cumulative Proportion of Total Counts')
    plt.title(f'Lorenz Curves for {", ".join([x["name"] for x in experiments])}')

    # for lorenz curve, we should also plot the line y = x
    plt.plot(np.linspace(0, 1, 500), np.linspace(0, 1, 500), label='y = x')
    plt.legend()
    plt.show()
    

def run_experiments(experiments):
    results = []
    for experiment_name, filters in experiments:
        data = read_data(filters)
        optimized_params = get_params(data)
        predictions, chi_square_stat, p_value = get_predictions(optimized_params, data)
        aggregate_stats = get_aggregate_stats(
            data, 
            optimized_params, 
            chi_square_results=(chi_square_stat, p_value)
        )
        mixing_points = get_mixing_parameters_points(optimized_params)
        lorenz_points = compute_lorenz_curve_points(optimized_params)
        
        results.append({
            'name': experiment_name,
            'params': optimized_params,
            'predictions': {
                'Count': predictions['Count'].tolist(),
                'Probs': predictions['Probs'].tolist(),
                'Predicted': predictions['Predicted'].tolist(),
                'Cumulative Probs': predictions['Cumulative Probs'].tolist(),
                'Data Probs': predictions['Data Probs'].tolist()
            },
            'aggregate_stats': aggregate_stats,
            'mixing_points': {
                'x': mixing_points[0].tolist(),
                'y': mixing_points[1].tolist()
            },
            'lorenz_points': {
                'x': lorenz_points[0].tolist(),
                'y': lorenz_points[1].tolist()
            }
        })

    # graph_predictions_vs_actual_for_multiple_completed_experiments(results)
    # graph_lorenz_curve_for_multiple_completed_experiments(results)
    # plot_mixing_parameters_points(results)
    return results

if __name__ == "__main__":
    experiments = [
        ("Male", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male']
        }),
        ("Male non-religious", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
            RELIGION_LOOKUP_KEY: RELIGION_OPTIONS['No Religion']
        }),
        ("Male protestant", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
            RELIGION_LOOKUP_KEY: RELIGION_OPTIONS['Protestant']
        }),
        ("Male catholic", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
            RELIGION_LOOKUP_KEY: RELIGION_OPTIONS['Catholic']
        }),
    ]

    experiments_emote_abuse = [
        ("Female Sex Abuse", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
            SEXABUSE_HISTORY_LOOKUP_KEY: SEXABUSE_HISTORY_OPTIONS['Was ever sexually abused']
        }),
        ("Female No Sex Abuse", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
            SEXABUSE_HISTORY_LOOKUP_KEY: SEXABUSE_HISTORY_OPTIONS['Was never sexually abused']
        }),
        # ("Female Baseline", {
        #     GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
        # }),
        # ("Female Emotional Abuse", {
        #     GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
        #     EMOTABUSE_BY_PARENTS_LOOKUP_KEY: EMOTABUSE_BY_PARENTS_OPTIONS['Was ever emotionally abused by parents']
        # }),
        # ("Female No Emotional Abuse", {
        #     GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
        #     EMOTABUSE_BY_PARENTS_LOOKUP_KEY: EMOTABUSE_BY_PARENTS_OPTIONS['Was never emotionally abused by parents']
        # }),
    ]

    male_vs_female_baseline = [
        ("Male Baseline", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
        }),
        ("Female Baseline", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
        }),
    ]

    no_mother_figure_experiment = [
        ("No Mother Figure Male", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
            NO_MOTHER_FIGURE_LOOKUP_KEY: NO_MOTHER_FIGURE_OPTIONS['No Mother Figure']
        }),
        ("No Mother Figure Female", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
            NO_MOTHER_FIGURE_LOOKUP_KEY: NO_MOTHER_FIGURE_OPTIONS['No Mother Figure']
        }),
        ("No mother figure overall", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS["Both"],
            NO_MOTHER_FIGURE_LOOKUP_KEY: NO_MOTHER_FIGURE_OPTIONS['No Mother Figure']
        }),
        ("All males", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
        }),
        ("All females", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
        }),
    ]

    no_father_figure_female_experiment = [
        ("No Father Figure Female", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
            NO_FATHER_FIGURE_LOOKUP_KEY: NO_FATHER_FIGURE_OPTIONS['No Father Figure']
        }),
        ("All females", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Female'],
        }),
    ]

    all_males_experiment = [    
        ("All males", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Male'],
        }),
    ]

    divorced_before_vs_never_divorced_experiment = [
        ("Divorced before", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Both'],
            SEPARATED_MARITAL_STATUS_LOOKUP_KEY: SEPARATED_MARITAL_STATUS_OPTIONS['Divorced']
        }),
        ("Currently Married", {
            GENDER_LOOKUP_KEY: GENDER_OPTIONS['Both'],
            MARITAL_STATUS_LOOKUP_KEY: MARITAL_STATUS_OPTIONS['Married']
        })
    ]



    # run_experiments(experiments_emote_abuse)
    # run_experiments(experiments)
    # run_experiments(all_males_experiment)

    # run_experiments(all_males_experiment)

    # loads(dumps(run_experiments(male_vs_female_baseline)))

    run_experiments(male_vs_female_baseline)



    


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
Current Marital status: MARSTAT
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

Demographic:
- How much does family income play a role?
- How much does education play a role?
- Racial splits
- gender reporting
- age reporting

(so much LMAO)

For each analysis:
1. Fit a spiked NBD @ 0
2. Compute mean, median, mode, variance
3. Get chi-square value for comparisons
4. Compare spike, r and alpha (High spike == very likely to be 0). Low r means more heterogenous (long tail) while
high r means more homogenous
5. For fun, see what percentile of people it would take to contribute to 80% of body count, and plot the lorenz curve
"""




