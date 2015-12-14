import sys
import data_reader

data_path = sys.argv[1]
data = data_reader.DataSet(data_path)
data_set = data.item_set

print 'LENGTH: {}'.format(data_set.count())
print 'NUMBER of STUDENTS: {}'.format(data_set.group_by(lambda x: x.user_name).count())
