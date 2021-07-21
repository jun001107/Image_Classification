import argparse
import train_function

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir', type=str, action = 'store')
parser.add_argument('--arch',type=str, dest = 'arch', default='vgg19', choices = ['vgg13', 'vgg19'])
parser.add_argument('--learning_rate', dest = 'learning_rate', default=0.001)   
parser.add_argument('--hidden_units', type=str, dest = 'hidden_units', default='512', help='Sizes for hidden layers. Seperate by comma if more then one')
parser.add_argument('--epochs', type=int, dest = 'epoch', default=9)
parser.add_argument('--gpu', type=bool, default=True, action = "store_true")
parser.add_argument('--save_dir', type=str, dest = 'save_dir', default='.', help='Path to where the checkpoint is stored to')
parser.add_argument('--output_units', type=int, dest = 'output_units', default=102, help='Output size of the network')

args = parser.parse_args()
train_function.train(args)