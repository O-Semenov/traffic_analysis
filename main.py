from nfstream import NFPlugin, NFStreamer
import numpy as np
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def score(train, test):
    return np.sum(train == test) / len(train)


def app_name_prediction(streamer, X):
    application_names = streamer["application_name"].unique()
    AN_indexs = dict(zip(application_names, list(range(0, len(application_names)))))
    y = np.array(streamer["application_name"])
    for ind in range(0, len(y)):
        y[ind] = AN_indexs[y[ind]]
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)

    model = KNeighborsClassifier(n_neighbors=len(application_names))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return score(y_test, pred)


def category_name_prediction(streamer, X):
    application_category = streamer["application_category_name"].unique()
    AC_indexs = dict(zip(application_category, list(range(0, len(application_category)))))
    y = np.array(streamer["application_category_name"])
    for ind in range(0, len(y)):
        y[ind] = AC_indexs[y[ind]]
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)

    model = KNeighborsClassifier(n_neighbors=len(application_category))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return score(y_test, pred)


def main():
    streamer = NFStreamer(source="main_test.pcap").to_pandas()

    X = np.array(streamer[["bidirectional_packets", "bidirectional_bytes"]])
    print(app_name_prediction(streamer, X))
    print(category_name_prediction(streamer, X))


if __name__ == '__main__':
    main()

'''

src_ip
dst_ip
protocol
ip_version
bidirectional_packets
bidirectional_bytes
application_name
application_category_name
content_type
'''
