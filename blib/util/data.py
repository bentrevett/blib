import torch.utils.data

class TwoOneDataset(torch.utils.data.Dataset):

    def __init__(self, input1, input2, output1):
        """
        The standard PyTorch dataset objects only handle single-input-single-output data loading
        This handles two-input-single-output
        """

        assert len(input1) == len(input2) == len(output1)

        self.input1 = input1
        self.input2 = input2
        self.output1 = output1


    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        #self.output[idx] needs to be wrapped in a list to handle single values
        return torch.LongTensor(self.input1[idx]), torch.LongTensor(self.input2[idx]), torch.LongTensor([self.output1[idx]])

class OneTwoDataset(torch.utils.data.Dataset):

    def __init__(self, input1, output1, output2):
        """
        The standard PyTorch dataset objects only handle single-input-single-output data loading
        This handles one-input-two-output
        """

        assert len(input1) == len(output1) == len(output2)

        self.input1 = input1
        self.output1 = output1
        self.output2 = output2


    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        #self.output[idx] needs to be wrapped in a list to handle single values
        return torch.LongTensor(self.input1[idx]), torch.LongTensor([self.output1[idx]]), torch.LongTensor([self.output2[idx]])

class TwoTwoDataset(torch.utils.data.Dataset):

    def __init__(self, input1, input2, output1, output2):
        """
        The standard PyTorch dataset objects only handle single-input-single-output data loading
        This handles two-input-two-output
        """

        assert len(input1) == len(input2) == len(output1) == len(output2)

        self.input1 = input1
        self.input2 = input2
        self.output1 = output1
        self.output2 = output2

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        #self.output[idx] needs to be wrapped in a list to handle single values
        return torch.LongTensor(self.input1[idx]), torch.LongTensor(self.input2[idx]), torch.LongTensor([self.output1[idx]]), torch.LongTensor([self.output2[idx]])