import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dimensionality_reduction import remove_missing
import os

def explore(st, df):
    st.write(f"The amount of datapoints in the dataset is: {len(df)}")
    st.write("")
    st.write(f"The shape of the dataset is: {df.shape}")
    st.write("")
    st.write("A quick peek at the data")
    st.write(df.head(10))
    st.write("")
    st.write("The datatypes")
    st.write(df.dtypes.astype(str))
    st.write("")
    st.write("Some basic statistics")
    st.write(df.describe())
    st.write("")
    st.write("So our label is Sepsis, how balanced is this dataset?")
    st.write(df['SepsisLabel'].value_counts(normalize=True))
    st.write("")
    st.write("What is the age distribution?")
    fig = plt.figure(figsize=(16, 6))
    sns.distplot(df['Age'])
    st.pyplot(fig)
    st.write("Very imbalanced, the mean is 62 as we already saw")
    st.write("")
    st.write("How are the genders distributed?")
    st.write(df['Gender'].value_counts(normalize=True))
    st.write("A bit more men")
    st.write("")
    st.write("Let's actually bin the ages and have a look at the gender to age distribution") # https://stackoverflow.com/questions/45273731/binning-a-column-with-python-pandas
    bins = [0, 18, 25, 40, 65, 100]
    df['age_bin'] = pd.cut(df['Age'], bins)
    st.write(df.groupby('age_bin')['Gender'].value_counts(normalize=True))
    st.write("So we have barely any boys and the older they get, the more we have men")
    st.write("")
    st.write("Do we see if any of these are more septic?")
    st.write("Let's see for the gender")
    st.write(df.groupby('Gender')['SepsisLabel'].value_counts(normalize=True))
    st.write("Nothing weird to see here")
    st.write("")
    st.write("And for the age bins?")
    st.write(df.groupby('age_bin')['SepsisLabel'].value_counts(normalize=True))
    st.write("No correlation here either")
    st.write("")
    st.write("Also it looked like we are missing a lot of values, let's have a look "
             "how many values we have for each column")
    missing = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)
    st.write(missing) # https://stackoverflow.com/questions/51070985/find-out-the-percentage-of-missing-values-in-each-column-in-the-given-dataset
    st.write("")
    st.write("Okay, that's really a lot, let's have a look for how many we have less than 61% of data")
    st.write(missing[missing > .39].count())
    # st.write(list(missing[missing > .39].index))
    st.write("36 Columns, that's a lot")
    st.write("Let's remove these columns and have a look if we see some correlation between the features")
    df = remove_missing(df, 0.61)
    st.write(df.columns)
    fig = plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Correlation Heatmap')
    st.pyplot(fig)
    st.write("MAP is okayish correlated with DBP and SBP")
    st.write("")
    st.write("Let's have a look at some scatter plots")
    image_name = "images/map_dbp.png"
    if os.path.isfile(image_name):
        st.image(image_name)
    else:
        fig = plt.figure(figsize=(16, 6))
        sns.scatterplot(data=df, x="MAP", y="DBP", hue="SepsisLabel", palette="viridis")
        fig.savefig(image_name)
        st.pyplot(fig)
    image_name = "images/map_sbp.png"
    if os.path.isfile(image_name):
        st.image(image_name)
    else:
        fig = plt.figure(figsize=(16, 6))
        sns.scatterplot(data=df, x="MAP", y="SBP", hue="SepsisLabel", palette="viridis")
        fig.savefig(image_name)
        st.pyplot(fig)
    st.write("They are however not correlated enough to be removing one feature")
    st.write("")
    st.write("Let's have a look at the Time Series")
    ts = df['patient'].value_counts().reset_index().rename(columns={'patient':'length'})
    ts.columns = ['patient','length']
    fig = plt.figure(figsize=(16, 6))
    sns.distplot(ts['length'])
    st.pyplot(fig)
    st.write(ts['length'].describe())
    st.write("Patients are mostly between 1 and 2 days hospitalized")