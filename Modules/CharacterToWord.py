from torch import nn
import torch
import copy


class CTW(nn.Module):
    """
    Character to word class
    """

    def __init__(self):
        super().__init__()
        self.algo = 0
        self.embeddings_size = [256, 320, 160]

    def seq(self, TensorInput):
        while TensorInput.shape != torch.squeeze(TensorInput, 0).shape:
            TensorInput = torch.squeeze(TensorInput, 0)
        return TensorInput

    def check_and_convert_to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            try:
                data = torch.tensor(data)
            except Exception as e:
                raise TypeError(f"Unable to convert to tensor. Error: {e}")
        data = self.seq(data)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        if len(data.shape) != 2 or data.shape[1] not in self.embeddings_size:
            raise ValueError(f"Tensor shape should be N*{self.embeddings_size}. Got {data.shape}.")

        return data

    def run(self, InputVector, wordGroupsID):
        Input = self.check_and_convert_to_tensor(InputVector)
        return self.Function0(Input, wordGroupsID)

    def Function0(self, InputVector, wordGroupsID):
        """
        InputVector_：输入的字向量
        wordGroupsID：分词结果ID
        Example：
        InputVector：[玛_Vector，丽_Vector，有_Vector，只_Vector，小_Vector，绵_Vector，羊_Vector]
        wordGroupsID：[[0,1], [2], [3], [4,6], [5,6]]
        """
        result = None
        for Groups in wordGroupsID:
            # print(Groups)
            result_tensor = None
            if len(Groups) == 1:
                Groups.append(Groups[0])
            for ID in range(Groups[0], Groups[1] + 1, 1):
                # print(f"InputVector:{InputVector}")
                # print(ID)
                # print(f"before:{result_tensor}")
                if result_tensor is None:
                    result_tensor = copy.deepcopy(InputVector[ID])
                    # 这里还需要深拷贝一下，不然InputVector就被加上去了
                else:
                    result_tensor += InputVector[ID]
                # print(f"InputVector[ID]:{InputVector[ID]}")
                # print(f"after:{result_tensor}")
            if result is None:
                result = result_tensor.unsqueeze(0)
            else:
                # result = torch.stack([result, result_tensor])
                # print("result:{}, result_tensor:{}".format(result.shape, result_tensor.shape))
                result = torch.cat((result, result_tensor.unsqueeze(0)), dim=0)

        return result


if __name__ == "__main__":
    x = CTW()
    tensor = torch.randn(1, 1, 5, 256)
    print(x.seq(tensor).shape)
    print(x.run(
        [[1 for i in range(256)], [2 for i in range(256)], [3 for i in range(256)], [4 for i in range(256)]],
        [[0], [1, 3], [0, 2]]
    ))
