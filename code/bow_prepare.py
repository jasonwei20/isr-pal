
from utils import *
import config

#generate word_to_idx
f_dict = {}
add_folder_to_f_dict(f_dict, join(config.dataset, 'data'))
word_to_idx = get_top_words(f_dict, 5000)

#save word_to_idx
output_path = join(config.dataset, "bow_word_to_idx.p")
pickle.dump(word_to_idx, open(output_path, 'wb'))
