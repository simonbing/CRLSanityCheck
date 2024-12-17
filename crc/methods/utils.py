from crc.methods import ContrastCRL, Multiview


def get_method(method):
    match method:
        case 'contrast_crl':
            return ContrastCRL
        case 'multiview':
            return Multiview
        case _:
            AssertionError(f'Undefined method {method}!')
