import sys
from bpnn import Nn
from linear import Linear
from rt import Rt
from svr import SvrN
from feature_selection import Selector

class Manager:
    provider = Selector()

    def nn(self, test_no):
        nn = Nn(self.provider)
        return nn.test(test_no)

    def linear(self, test_no):
        linear = Linear(self.provider)
        return linear.test(test_no)

    def tree(self, test_no):
        tree = Rt(self.provider)
        return tree.test(test_no)

    def svr(self, test_no):
        svr = SvrN(self.provider)
        return svr.test(test_no)

# Pagrindinė prgorama
class Main:
    def __init__(self, argv):
        self.argv = argv

    def execute(self):
        modes = ['nn', 'linear', 'tree', 'svr', 'params']
        manager = Manager()
        usage = 'Usage: %s %s [output file] [1|2 (test no)]' % (self.argv[0], '|'.join(modes))

        if len(self.argv) < 2:
            raise ValueError(usage)

        mode_name = self.argv[1]
        test_no = 1
        if (len(self.argv) == 4):
            test_no = int(self.argv[3])

        if 'nn' == mode_name:
            results = manager.nn(test_no)
        elif 'linear' == mode_name:
            results = manager.linear(test_no)
        elif 'tree' == mode_name:
            results = manager.tree(test_no)
        elif 'svr' == mode_name:
            results = manager.svr(test_no)
        elif 'params' == mode_name:
            results = ""
            results += 'Raw data inputs: {0:d}'.format(manager.provider.provider.getInputCount())
            results += '\nFeatures selected: {0:d}'.format(manager.provider.outputs)
            results += '\nFeatures:'
            for feature in manager.provider.sortedFields:
                results += '\n{0:s}: {1:f}'.format(feature[1], feature[0])
        else:
            raise ValueError(usage + '\nUnrecognised mode: ' + mode_name)

        if (len(self.argv) > 2 and self.argv[2] != ''):
            w = open(self.argv[2], 'w')
            for line in results:
                w.write('%f;%f\n' % (line[0], line[1]))
            w.close()
        else:
            print(results)

if __name__ == '__main__':
    try:
        main = Main(sys.argv)
        main.execute()
    except Exception as ex:
        print(ex)