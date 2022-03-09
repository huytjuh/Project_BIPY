### AB TESTING ###

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# NORMALITY
def shapiro_normality(A, B, p=0.05):
    shapiro_A = shapiro(A)[1] < p
    shapiro_B = shapiro(B)[1] < p
    if (shapiro_A == False) & (shapiro_B == False):
        bool_output, bool_str = True, 'H0 not rejected'
    else:
        bool_output, bool_str = False, 'H0 rejected'

    print('Test for Normality \nH0: Distribution Normal \nH1: Distribution not Normal \nResult: {} (p_values=({:.3f}, {:.2f}))'.format(bool_str, shapiro(A)[1], shapiro(B)[1]))
    return bool_output

def levene_homogeneity(A, B, p=0.5):
    levene_test = levene(A, B)[1] < p

    if levene_test == False:
        bool_output, bool_str = True, 'H0 not rejected'
    else:
        bool_output, bool_str = False, 'H0 not rejected'

    print('Test for Homogeneity \nH0: Homogeneity \nH1: Heterogeneity \nResult: {} (p_value={:.3f})'.format(bool_str, levene(A, B)[1]))
    return bool_output
  
def AB_test(A, B, p=0.5):
    if shapiro_normality(A, B) == True:
        if levene_homogeneity(A, B) == True:
            AB_output, method = ttest_ind(A, B, equal_var=True)[1], 'Parametric & Homogeneity'
        else:
            AB_output, method = ttest_ind(A, B, equal_var=False)[1], 'Parametric & Heterogeneity'
    else:
        AB_output, method = mannwhitneyu(A, B)[1], 'Non-Parametric'

    if AB_output < p:
        bool_output, bool_str = True, 'H0 rejected'
    else:
        bool_output, bool_str = False, 'H0 not rejected'
      
    print('{} Test: \nH0: A == B \nH1: A != B \nResult: {} (p_value={:.3f})'.format(method, bool_str, AB_output))
    return bool_output
