import pickle

train_data = pickle.load(open('data/gerrit/gerrit_test_cc2ftr.pkl', 'rb'))
train_ids, train_labels, train_messages, train_codes = train_data