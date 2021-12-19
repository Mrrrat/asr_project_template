# Don't forget to support cases when target_text == ''

import editdistance

def calc_wer(target_text, predicted_text) -> float:
    t_words = target_text.split()
    p_words = predicted_text.split()
    if len(t_words) == 0 and len(p_words) != 0:
        return 1.0
    return editdistance.eval(t_words, p_words) / len(t_words)


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0 and len(predicted_text) != 0:
        return 1.0
    return editdistance.eval(target_text, predicted_text) / len(target_text)
