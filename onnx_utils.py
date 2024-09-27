import torch
import torch.nn as nn
from torch.utils import data
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
from math import isclose
import onnx


def convert_to_onnx(model: nn.Module, dummy: torch.Tensor, device: torch.device, path: str) -> None:
    model.eval()
    model.to(device)

    torch.onnx.export(model,               
                    dummy,
                    path, 
                    export_params=True,        
                    input_names = ['input'],   
                    output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},
                                 'output' : {0 : 'batch_size'}})
    

def check_onnx_model(torch_model: nn.Module, 
                     onnx_path: str, 
                     dummy: torch.Tensor,
                     dataloaders: dict[str, data.DataLoader],
                     torch_val_loss: float,
                     torch_val_acc: float
                    ) -> None:
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    torch_out = torch_model(dummy)

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_outs = ort_session.run(None, {'input': to_numpy(dummy)})

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print('Exported model has been tested with ONNXRuntime, and the result looks good!')

    criterion = nn.CrossEntropyLoss() 

    running_loss = 0
    running_corrects = 0 
    amount = 0

    for inputs, targets in (pbar := tqdm(dataloaders['val'], desc='Starting ONNX model evaluation.')):
            
        logits = ort_session.run(None, {'input': to_numpy(inputs)})[0]
        
        preds = np.argmax(logits, axis=1)
        
        loss = criterion(torch.tensor(logits), targets)
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += np.sum(preds == to_numpy(targets))

        amount += inputs.size(0)

        val_loss = running_loss / amount
        val_accuracy = running_corrects.item() / amount
        
        pbar.set_description(f'ONNX Eval | Val Loss: {val_loss:.06f} | Val acc: {val_accuracy:.06f}')

    print(val_loss, torch_val_loss)
    print(val_accuracy, torch_val_acc)

    assert isclose(val_loss, torch_val_loss, rel_tol=1e-5), 'ONNX validation loss is different from torch validation loss'
    assert isclose(val_accuracy, torch_val_acc, rel_tol=1e-5), 'ONNX accuracy is different from torch accuracy'

    print('All metrics match!')