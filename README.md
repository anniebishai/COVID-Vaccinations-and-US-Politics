# Disparities in COVID Vaccination Across the American Political Divide

### A data science investigation by Ankita Patil, Jacqueline Simeone, Andy Orfalea, and Annie Bishai

---

# Abstract

The COVID-19 pandemic emerged at a time of intense political division in the United States. The election of Donald Trump, a [denier](https://www.theatlantic.com/politics/archive/2016/12/donald-trump-climate-change-skeptic-denial/510359/) of scientific fact, in 2016 was a symptom of deep social, ideological, and informational divides and a herald of intensifying polarization on all of these axes. When voters had the choice of re-electing Trump or electing Joe Biden, the Delta wave of the virus had just peaked. Election Day 2020 was concurrent with the national COVID [death tally](https://covidtracking.com/data/national/deaths) officially surpassing 225,000. Election disinformation, much of it perpetuated by the Trump administration, was a [major issue](https://misinforeview.hks.harvard.edu/article/covid-19-misinformation-and-the-2020-u-s-presidential-election/) in the election, according to experts. Although the Biden administration has taken a drastically different approach to public information and policy, vaccination acceptance as well as COVID death rates continue to be stratified [along political lines](https://www.nytimes.com/2021/09/27/briefing/covid-red-states-vaccinations.html).

Even the reddest states have blue counties, and vice versa, and the COVID pandemic has highlighted that these local divides manifest in a consequential way. Maps readily [accessible](https://www.nytimes.com/interactive/2021/us/covid-cases.html?name=styln-coronavirus&region=TOP_BANNER&block=storyline_menu_recirc&action=click&pgtype=Interactive&variant=0_Control&is_new=false) from media outlets reveal local disparities in COVID incidence, vaccination, and other measures. Moreover, policy pertaining to COVID is often determined on municipal and county levels, and the proliferation of [protests](https://bethesdamagazine.com/bethesda-beat/schools/hundreds-of-mcps-students-walk-out-in-protest-of-districts-handling-of-covid-19/) [against various](https://spectrumlocalnews.com/nys/buffalo/news/2021/12/01/group-plans-covid-19-mandate-protest) [COVID](https://www.latimes.com/california/story/2021-11-29/la-me-california-covid-mandates-constitutional-republic) [policies](https://www.kcra.com/article/stanislaus-county-students-parents-protest-covid-19-vaccine-mandate/37888350#) in recent months reveals that on the most local levels Americans are of diverse mindsets. The goal of this project is to explore social and political divides on a more granular level than the red state/blue state binary, using additional social and demographic variables to illuminate, and examine the strength of, these divides.

# Hypothesis

We hypothesize that applying a KMeans clustering model to our vaccine- and vaccine hesitancy-related data will reveal stratification within states along similar lines to those seen in the 2020 presidential election. We predict that with the inclusion of demographic and socioeconomic data not directly related to COVID, this model will generally continue to reflect those patterns. This hypothesis is informed by the fact of missing data, which may limit the model's ability to discern other meaningful differences among counties.

Because of the close vaccine/election relationships in these data we have observed--and hope to observe further through our clustering model--we additionally predict that applying a supervised machine learning model to the same data may yield an accurate prediction of the voting tendencies of a county. We will explore this by training the model with a target variable drawn from the 2020 presidential election results.


# Data 

## Sources

- Vaccination administration progress: [CDC](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh) We used March 1, 2021, September 1, 2021, and March 1, 2022 as "checkpoints" in rates of first dose administration, series completeion, and booster administration. 
- Census information: Included in the above CDC dataset; original source is the [2019 Vintage Census Population Estimates](https://www.census.gov/newsroom/press-kits/2020/population-estimates-detailed.html).
- Social vulnerability and uninsurance data (2018): [Agency for Toxic Substances and Disease Registry (ATSDR)](https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html)
- Vaccine hesitancy data: [Institute for Health Metrics and Evaluation (IHME)](https://www.healthdata.org/acting-data/covid-19-vaccine-hesitancy-us-county-and-zip-code), measured Feb. 18, 2022 through Feb. 24, 2022. [(Complete Methodology)](https://www.healthdata.org/node/8770)
    * Hesitancy measures are calculated based on response to the question, "If a vaccine to prevent COVID-19 (coronavirus) were offered to you today, would you choose to get vaccinated?" by respondents who indicate they have not already received a vaccine. Our hesitancy measures for a given county are based on the mean of zip-code-level predictions by IHME.
- 2020 election results: [Ajaypal Singh on Kaggle](https://www.kaggle.com/ajaypalsinghlo/us-election-2020-president-county-candidate/data?select=president_county_candidate.csv)

## Missing Data and Caveats

- In the CDC vaccine-tracking dataset, which contains data reported by vaccine providers, 469 of 3138 counties were missing values for one or more of the dates we sampled. Some causes of missing or incomplete dataa are listed [here](https://www.cdc.gov/coronavirus/2019-ncov/vaccines/distributing/about-vaccine-data.html), but we were not able to find specific reasons for the non-reporting of every county. Based on the distribution of data, we ascertained that the data was not missing at random. A substantial portion of the missing data (74% of the missing values on one of the three dates used) are from counties in Texas, where there is known and extreme [partisanship](https://www.texasattorneygeneral.gov/news/releases/texas-vs-fed-er-al-vac-cine-mandates) around federal COVID policy. Considering this and the fact that the missing vaccination data comprised over 10% of the total observations, we chose not to omit these rows from the analysis or modeling. They are represented as 0.0 in the data.

- In order to minimize the impact of missing data, we created a column in which zero (missing) values from the September 2021 vaccination columns are imputed as the mean of March 2021 and March 2022 values. We also added a binary feature for each vaccination checkpoint date in which observations were "flagged" as missing or not. Imputed September values were used in the clustering model along with the "missing" feature, thus mathematically differentiating rows that were imputed from those that contain real vaccination data.

- Alaska's election districts to not correspond to its counties, so county-wide data from the 2020 election is not included here. Rather, the values for election data in all Alaska counties reflect a state average.

## Data Dictionary

**[dataset as .csv file](./clean_data/final_data_formodeling.csv)**

| Feature | Type | Description |
|---------|------|--------|
| County_State | string | County and state |
| Administered_Dose1_Recip_030121, 090121, 030122 | float | Number of people residing in the county who have received at least one dose of a covid vaccine, reported on March 1, 2021, September 1, 2021, and March 1, 2022 |
| Administered_Dose1_Pop_Pct_030121, 090121, 030122 | float | Percent of Total Pop with at least one dose of a covid vaccine, reported on March 1, 2021, September 1, 2021, and March 1, 2022 |
| Dose1_Pop_Pct_090121_impute | float | Same as Dose1_Pop_Pct_090121, but with 0s from non-reporting counties replaced with the mean of that county's March 2021 and March 2022 data |
| Series_Complete_Pop_Pct_090121, 030122 | float | Percent of people who are fully vaccinated (have second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives, reported on September 1, 2021 and March 1, 2022 |
| Series_Complete_Pop_Pct_090121_impute | float | Same as Series_Complete_Pop_Pct_090121, but with 0s from non-reporting counties replaced with the mean of that county's March 2021 and March 2022 data |
| Booster_Doses_Vax_Pct_030122 | float | Percent of people who are fully vaccinated and have received a booster (or additional) dose of a covid vaccine, reported on March 1, 2022 |
| Missing_March21 | integer | 0 = county reported vaccine data on March 1, 2021; 1 = county did not report, and value is 0
| Missing_Sept21 | integer | 0 = county reported vaccine data on Sept 1, 2021; 1 = county did not report, and value is 0
| Missing_March22 | integer | 0 = county reported vaccine data on March 1, 2022; 1 = county did not report, and value is 0
| Pct_Uninsured | float | Percent of people without health insurance |
| SVI_Socio | float | Mean social vulnerability percentile, Socioeconomic theme |
| SVI_HHDisab | float | Mean social vulnerability percentile, Household Composition & Disability theme |
| SVI_Minority | float | Mean social vulnerability percentile, Minority Status & Language theme |
| SVI_HousingTransp | float | Mean social vulnerability percentile, Housing Type & Transportation theme |
| SVI_Overall | float | Mean social vulnerability percentile, overall |
| SVI_Ctgy | float | Quartile in which the county falls in overall SVI score |
| Metro_Status | float | 0 = non-metro, 1 = metro
| Census2019 | float | Total population |
| Census2019_18to64Pop | float | Population aged 18-64 |
| Census2019_65PlusPop | float | Population aged 65 and over |
| Census2019_Pct_65Plus | float | Population aged 65 and over as a percent of total population | 
| Pct_Hesitant_Feb22 | float | Mean population percentage responding that they would "probably," "probably not," or "definitely not" take a Covid vaccine |
| Pct_Somewhat_Hesitant_Feb22 | float | Mean population percentage responding that they would "probably" or "probably not" take a Covid vaccine |
| Pct_Highly_Hesitant_Feb22 | float | Mean population percentage responding that they would "definitely not" take a Covid vaccine |
| Hesitant_nzip | float | Hesitancy percentage (mean) times number of zip code districts from which mean was taken |
| Somewhat_Hesitant_nzip | float | Somewhat-hesitant mean times number of zip code districts from which mean was taken |
| Highly_Hesitant_nzip | float | High-hesitancy mean times number of zip code districts from which mean was taken |
| Candidate_Won | float | 0 = Joe Biden; 1 = Donald Trump |
| Pct_Trump | float | Votes for Trump as a percent of total votes |
| Pct_Biden | float | Votes for Biden as a percent of total votes |

# Modeling 

## KMeans Clustering

The KMeans model fit the data best with three clusters. The features used in the model, and the mean of each feature by cluster (i.e. the centroid locations), are displayed here:

| Cluster | 0 | 1 | 2 |
|---------|---|---|---|
| Pct_Hesitant_Feb22 | 0.293 | 0.176 | 0.252 |
| Pct_Somewhat_Hesitant_Feb22 | 0.083 | 0.050 | 0.074 |
| Pct_Highly_Hesitant_Feb22 | 0.210 | 0.126 | 0.178 |
| Dose1_Pop_Pct_030121 | 13.699 | 15.968 | 2.108 |
| Dose1_Pop_Pct_090121_impute | 39.939 | 56.148 | 19.306 |
| Dose1_Pop_Pct_030122 | 50.963 | 70.741 | 43.620 |
| Series_Complete_Pop_Pct_090121_impute | 34.221 | 50.031 | 21.085 |
| Series_Complete_Pop_Pct_030122 | 4.706 | 62.176 | 41.196 |
| Booster_Doses_Vax_Pct_030122 | 41.326 | 46.515 | 34.677 |
| Missing_Sept21 | 0.002 | 0.037 | 0.619 |
| SVI_Overall | 0.583 | 0.459 | 0.586
| Metro_Status | 0.215 | 0.599 | 0.329
| Census2019_Pct_65Plus | 0.202 | 0.191 | 0.197 |


**Summary of observations from clustering model:**

*Cluster 0*

- These counties have the highest vaccine hesitancy, but their vaccination rates are in the middle of the pack. This could be due to Cluster 2's vaccination averages being brought down by non-reporting.
- These counties are generally rural (metro status=0); their social vulnerability is high as measured by socioeconomic status and about the same overall as Cluster 2's.
- Trump won the election in 94.8% of these counties, the highest percent among all the clusters.
- Overall this cluster appears demographically and politically similar to Cluster 2.

*Cluster 1*

- These counties have the lowest hesitancy and highest vaccination rates overall.
- These counties are most urban (metro status=1) on average and comprise the largest total population (about 5x the population of Cluster 0 and 8x the population of Cluster 2).
- These counties have the lowest social vulnerability rankings overall and by socioeconomic status, but they contain more minority population than Cluster 0.
- These counties had the lowest rates of voting for Trump, with Trump victory occurring in 64% of counties compared to 83% nationwide.

*Cluster 2*

- Most counties that did not report vaccination data are in this cluster. Nearly half of these counties are in Texas.
- Possibly due to non-reporting, these counties' vaccination rates are the lowest. The hesitancy rates (17.8% on average are highly hesitant), however, support the hypothesis that vaccination efforts have in fact faced barriers in these counties.
- These counties have the highest social based on minority status, but comparable overall social vulnerability to Cluster 0.
- These counties had an above-average rate of Trump election victory, but not as high a rate as Cluster 0.


## Classification Model

*By Andy Orfalea*

To further investigate the relationship between the COVID vaccine and politics, and because the KMeans clustering model produced clusters with distinct differences in voting patterns without using this as a feature, we wanted to see if we could predict 2020 county election results using COVID vaccine data. In predicting this, our model primarily used COVID vaccine statistics but we also added two county demographic stats we felt were very relevant to COVID-- "Percent of Population over 65 years old" and "SVI Category".

Because Trump won 83% of all counties, the baseline model would be 83% accurate and that’s what we set out to beat.  After trying multiple models, we landed on a Stacking model that ensembled predictions from four different classifying models (K-Nearest Neighbors, Random Forest, Bagging, and Ridge). The Stacking model took predictions from all of those models, and used Logistic Regression to make its ultimate predictions. We ended up with an accuracy of 90.3% so we were able to improve on the baseline model by more than 7%.

# Conclusions

Our KMeans clustering model was partially successful in lending insight to the relationships among demographic, social vulnerability, and vaccine-related factors in the United States, and the relationship of these factors to election outcomes in 2020. We selected features for which the data was best modeled in three clusters, which necessarily lent insight into patterns of stratification with more detail than the red-state/blue-state divide. The three clusters that the model found did show marked differences in their 2020 election outcomes, supporting this portion of our hypothesis. 

However, as we foresaw, the clustering model suffered from missing data bearing a conceivable relationship (causal or not) to politics. Interestingly, counties that did not report data at one or more points in time were grouped into a single cluster (Cluster 2) which was otherwise relatively similar to another (Cluster 0). It is likely that this model is overfitted to this particular moment in time, and that the information from our clusters should not be the basis for generalization.

The stacked classification model successfully predicted 2020 election results with a higher degree of accuracy than the baseline using vaccination-related variables. This further supports our hypothesis that these variables are closely related. However, the fact that the vaccination data contained many missing values and that these values were represented as zeroes suggests that caution should be taken in interpreting specific numeric relationships regarding vaccine-related observations.

The outcomes of both modeling exercises generally reinforce what has been observed about politics and COVID vaccination, and they allow some insight into specific trends and variable relationships as seen in the notebooks and slides contained in this repo. In addition to potential usefulness of this investigation to public health officials and government at the state and federal levels, the project is also of use to the general public, as a snapshot of where we were and where we are, and the extent to which the politics of two years ago are still largely replicated in our pandemic response.

---

*README by Annie Bishai*

*Slide deck by Ankita Patil, Jacqueline Simeone, & Andy Orfalea*