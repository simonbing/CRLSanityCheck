from crc.methods import ContrastCRL, MultiviewIv


def get_method(method):
    match method:
        case 'contrast_crl':
            return ContrastCRL
        case 'multiview_iv':
            return MultiviewIv

        case _:
            AssertionError(f'Undefined method {method}!')
