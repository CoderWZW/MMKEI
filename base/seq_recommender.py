from base.recommender import Recommender
from data.sequence import Sequence
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
from util.sampler import next_batch_sequence_for_test
from util.logger import Log
import sys
import numpy as np

class SequentialRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(SequentialRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Sequence(conf, training_set, test_set)
        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.max_len = int(self.config['max_len'])
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.feature=self.config['feature']

    def print_model_info(self):
        super(SequentialRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (sequence number: %d, item number %d)' % (self.data.raw_seq_num,self.data.item_num))
        #print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def predict(self,  seq,pos,seq_len,time_stamps_seq):
        return -1

    def item_mask_for_pop(self, pop, seq):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []

        for i, s in enumerate(seq):
            to_be_masked = np.where(pop[seq[i]] < 10 )
            masked[i, to_be_masked] = 1
            # print("masked",masked)
            labels = labels + list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = 0
        return augmented_seq, masked, np.array(labels)
    def test(self,pop):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        for n, batch in enumerate(next_batch_sequence_for_test(self.data, self.batch_size,self.labelgap,max_len=self.max_len)):
            seq, pos, seq_len,gap_batch = batch
            # aug_seq, masked, labels=self.item_mask_for_pop(pop,seq)
            seq_names = [seq_full[0] for seq_full in self.data.original_seq[n*self.batch_size:(n+1)*self.batch_size]]
            # if(n==1):
            #    print('',seq_names)
            #进行预测来找到候选物品
            # candidates = self.predict(aug_seq,seq, pos, seq_len,masked)
            candidates = self.predict( seq, pos, seq_len,gap_batch)

            for name,res in zip(seq_names,candidates):

                ids, scores = find_k_largest(self.max_N, res)
                item_names = [self.data.id2item[iid] for iid in ids if iid!=0 and iid<=self.data.item_num]
                #按照名字排序
                rec_list[name] = list(zip(item_names, scores))
            if n % 100 == 0:
                process_bar(n, self.data.raw_seq_num/self.batch_size)
        process_bar(self.data.raw_seq_num, self.data.raw_seq_num)
        # print(len(rec_list))
        return rec_list
    #
    # def evaluate(self, rec_list):
    #     self.recOutput.append('Sequence Id: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
    #     for user in self.data.test_set:
    #         line = user + ':'
    #         for item in rec_list[user]:
    #             line += ' (' + item[0] + ',' + str(item[1]) + ')'
    #             if item[0] in self.data.test_set[user]:
    #                 line += '*'
    #         line += '\n'
    #         self.recOutput.append(line)
    #     current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
    #     # output prediction result
    #     out_dir = self.output['-dir']
    #     file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
    #     FileIO.write_file(out_dir, file_name, self.recOutput)
    #     print('The result has been output to ', abspath(out_dir), '.')
    #     file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
    #     self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
    #     self.model_log.add('###Evaluation Results###')
    #     self.model_log.add(self.result)
    #     FileIO.write_file(out_dir, file_name, self.result)
    #     print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def evaluate(self, rec_list):
        return 0


    def fast_evaluation(self, epoch,pop):
        print('Evaluating the model...')

        rec_list = self.test(pop)
        measure,hr,NDCG = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        # bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + ' | '
        #bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        if epoch==self.maxEpoch-1:
            self.model_log.add('*Best Performance* ')
            self.model_log.add('Epoch:'+str(self.bestPerformance[0]) + ','+ bp)
       
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure,hr,NDCG
