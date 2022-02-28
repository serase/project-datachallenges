import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.interval_based import TimeSeriesForestClassifier, RandomIntervalSpectralEnsemble
from sklearn.metrics import f1_score

def prepare_ts(df, x, y, n):
    df.dropna(inplace=True)
    df_ts = df.sort_index().groupby(['patient',y])[x].apply(np.array).reset_index()
    df_ts["size"] = df_ts[x].apply(lambda x: len(x))
    df_ts = df_ts.query(f'size >= {n}')
    df_ts[x] = df_ts[x].apply(lambda x: x[:n+1])
    df_ts = df_ts[[y, x]]
    df_ts.columns = ['y','x']
    X = df_ts['x'].apply(pd.Series).values
    y = df_ts['y']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, stratify=y)
    return X_train, X_test, y_train, y_test

def run_ts(st, df):
    st.write("The data looks like this:")
    st.write(df.head(10))
    df.sort_index(inplace=True)
    st.header("Analyzing the time series")
    st.write("So do the time series lengths differ?")
    st.write(df.groupby('patient').count().describe()['HR'])
    st.write("Yes actually they do differ a lot")
    st.write("When do we get 95% of all the time series?")
    st.write(df.groupby('patient').count().SepsisLabel.quantile(0.1))
    st.write("")
    st.write("Let's look at the data for one person")
    st.write(df.query("patient == 'p014977'"))
    st.write("")
    st.header("Prepping the time series")
    st.write("Now we would create timeseries of the same length for both metrics, with length 8 we will catch all timeseries")
    st.write("With length 18 we would catch 95% and have more than double the size")
    st.write("HR")
    st.write("To prepare the data a few problems occurred: Grouping the correct timeseries together, getting it in the right format for the classifiers")
    X_train, X_test, y_train, y_test = prepare_ts(df, "HR", "SepsisLabel", 8)
    st.write("Let's visualize a time series: the heart rate of 100 patients")
    st.line_chart(X_train[0:101].T)

    st.header("Choosing a library")
    st.write("We chose sktime since it has the clearest documentation and we have the most experience with this library")
    st.write("Also it has the same syntax as sklearn and therefore it's easy to use")
    st.write("Let's try the TimeSeriesForestClassifier")
    st.write("Had some issues with sktime though")
    st.write("Firstly: C++ Compiler Build was failing")
    st.write("Secondly: Some absolute path error")
    st.write("Then I installed using anaconda and the problems were gone")
    st.write("")

    st.header("Evaluation Metric")
    st.write("We use the F1 Score since the classes are imbalanced and there is a serious downside to predicting false negatives!")
    st.write("")
    st.write("A little more on the F1 Score:")
    st.write("F1 = 2 * (precision * recall) / (precision + recall)")
    st.write("The F1 score can be interpreted as a harmonic mean of the precision and recall")
    st.write("The recall tells us how complete the result is (true positives / all positives)")
    st.write("The precision tells us how accurate the result is (true positives / predicted positives)")
    st.write("")

    st.header("TS Forest Classifier")
    st.write("Let's use the TS Forest Classifier length 8")
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.write(f1_score(y_test, y_pred))
    st.write("What would happen if we had a size of 18 for the time series?")
    X_train, X_test, y_train, y_test = prepare_ts(df, "HR", "SepsisLabel", 18)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.write(f1_score(y_test, y_pred))

    st.header("RISE")
    classifier = RandomIntervalSpectralEnsemble(min_interval=3, acf_lag=18, n_jobs=-1)
    st.write("For time series of length 8")
    X_train, X_test, y_train, y_test = prepare_ts(df, "HR", "SepsisLabel", 8)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.write(f1_score(y_test, y_pred))

    st.write("For time series of length 18")
    X_train, X_test, y_train, y_test = prepare_ts(df, "HR", "SepsisLabel", 18)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.write(f1_score(y_test, y_pred))
    return df
