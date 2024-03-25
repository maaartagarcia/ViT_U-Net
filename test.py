from utils import conv_mask_gt, transform_masks, transform_predictions, validate_input_images
import numpy as np
import pdb, os

def test_transform_masks_zeros():
    z = np.zeros((256, 256))

    result = transform_masks(z)

    assert result.shape == (256, 256, 2)
    assert np.all(result[:,:,0] == 1)
    assert np.all(result[:,:,1] == 0)

def test_transform_masks_ones():
    z = np.ones((256, 256))

    result = transform_masks(z)

    assert result.shape == (256, 256, 2)
    assert np.all(result[:,:,0] == 0)
    assert np.all(result[:,:,1] == 1)

def test_transform_predictions():
    prediccion_ejemplo = np.zeros((256, 256, 2))
    # Establecer algunos valores para la clase 1 (Manipulada) en la predicción de ejemplo
    prediccion_ejemplo[100:150, 100:150, 1] = 0.8  # Clase 1 (Manipulada) con alta probabilidad
    prediccion_ejemplo[200:220, 200:220, 1] = 0.3  # Clase 1 (Manipulada) con baja probabilidad
    # Transformar la predicción de ejemplo a una máscara binaria
    mascara_resultante = transform_predictions(prediccion_ejemplo, 0.5)
    # Verificar que los píxeles en la máscara resultante coinciden con la transformación esperada
    assert mascara_resultante.shape == (256, 256)
    assert np.all(mascara_resultante[100:150, 100:150] == 1), "Error: Los píxeles de la clase 1 no coinciden en la máscara resultante"
    assert np.all(mascara_resultante[200:220, 200:220] == 0), "Error: Los píxeles de la clase 0 no coinciden en la máscara resultante"

test_transform_masks_zeros()
test_transform_masks_ones()
test_transform_predictions()




