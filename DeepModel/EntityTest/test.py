from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from data import build_corpus_to_predict
from evaluating import Metrics
# from evaluate import ensemble_evaluate

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记


def main():
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # print("加载并评估hmm模型...")
    # hmm_model = load_model(HMM_MODEL_PATH)
    # hmm_pred = hmm_model.test(test_word_lists,
    #                           word2id,
    #                           tag2id)
    # metrics = Metrics(test_tag_lists, hmm_pred, remove_O=REMOVE_O)
    # metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    # metrics.report_confusion_matrix()  # 打印混淆矩阵

    # # 加载并评估CRF模型
    # print("加载并评估crf模型...")
    # crf_model = load_model(CRF_MODEL_PATH)
    # crf_pred = crf_model.test(test_word_lists)
    # metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()

    # # bilstm模型
    # print("加载并评估bilstm模型...")
    # bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    # bilstm_model = load_model(BiLSTM_MODEL_PATH)
    # bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    # lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
    #                                                bilstm_word2id, bilstm_tag2id)
    # metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()

    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    # ensemble_evaluate(
    #     [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
    #     test_tag_lists
    # )

def pred_BiLSTM_CRF():
    
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    
    # pred_word_lists, pred_tag_lists = build_corpus_to_predict("pred", make_vocab=False)

    pred_sentence = '有点奇怪的模型，似乎识别效果比较差'

    pred_word_lists = [[]]
    pred_tag_lists = [[]]

    for i in range(0, len(pred_sentence)):
        pred_word_lists[0].append(pred_sentence[i])
        pred_tag_lists[0].append('O')
        pass
    
    # print(pred_word_lists)
    # print(pred_tag_lists)


    # print('word2id------------------')
    # print(word2id)
    # print('tag2id-------------------')
    # print(tag2id)

    print("加载bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    
    # print('crf_word2id------------------')
    # print(crf_word2id)
    # print('crf_tag2id-------------------')
    # print(crf_tag2id)
    
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    
    pred_word_lists, pred_tag_lists = prepocess_data_for_lstmcrf(
        pred_word_lists, pred_tag_lists, test=False
    )
    
    
    # print(test_word_lists)
    # print(test_tag_lists)
    
    print('开始预测')
    
    lstmcrf_pred, target_tag_list = bilstm_model.test(pred_word_lists, pred_tag_lists,
                                                      crf_word2id, crf_tag2id)
    
    # metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()

    print('-------预测结果-----------')
    print(pred_word_lists)
    print(lstmcrf_pred)
    
    for i in range(0, len(pred_word_lists[0]) - 1):
        print(pred_word_lists[0][i] + ' '+ lstmcrf_pred[0][i])
        pass

    
    # print(target_tag_list)


if __name__ == "__main__":
    # main()
    
    pred_BiLSTM_CRF()
