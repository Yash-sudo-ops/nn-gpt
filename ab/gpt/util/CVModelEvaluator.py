
import os
import ab.nn.api as api
from ab.nn.util.Util import read_py_file_as_string

class CVModelEvaluator:
    def __init__(self, model_source_package: str, task='img-classification', dataset='cifar-10', metric='acc', prm=None, save_to_db=False, prefix = None, save_path = None):
        """
        Evaluates a given model on a specified dataset for classification
        :param model_source_package: The package name of the model to evaluate
        :param task: The task to evaluate the model on
        :param dataset: The dataset to evaluate the model on
        :param metric: The metric to evaluate the model on
        :param prm: The parameters to evaluate the model on
        :param save_to_db: Whether to save the results to the database
        """
        if prm is None:
            prm = {'lr': 0.01, 'batch': 10, 'dropout': 0.2, 'momentum': 0.9,
                   'transform': 'norm_256_flip', 'epoch': 1}
        self.model_package = model_source_package
        self.task = task
        self.dataset = dataset
        self.metric = metric
        self.prm = prm
        self.save_to_db = save_to_db
        self.prefix = prefix
        self.save_path = save_path

    # def evaluate(self, num_epochs, batch_size=4):
    #     train_loader = torch.utils.data.DataLoader(
    #         self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    #     )
    #     test_loader = torch.utils.data.DataLoader(
    #         self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    #     )

    #     self.model.to(self.device)

    #     criterion = torch.nn.CrossEntropyLoss().to(self.device)
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9)

    #     print("Training", self.model_package, "on", self.device)
    #     time.sleep(0.5)
    #     for _ in tqdm(range(num_epochs)):
    #         for i, data in enumerate(train_loader):
    #             inputs, label = data
    #             assert isinstance(inputs, torch.Tensor)
    #             assert isinstance(label, torch.Tensor)
    #             inputs, label = inputs.to(self.device), label.to(self.device)

    #             optimizer.zero_grad()
    #             output = self.model(inputs)
    #             loss = criterion(output, label)
    #             loss.backward()
    #             optimizer.step()

    #             del inputs
    #             del label
    #     print("Finished Training for", self.model_package)

    #     total = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data in test_loader:
    #             image, label = data
    #             assert isinstance(image, torch.Tensor)
    #             assert isinstance(label, torch.Tensor)
    #             image, label = image.to(self.device), label.to(self.device)

    #             outputs = self.model(image)
    #             # Using the highest energy as output
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += label.size(0)
    #             correct += (predicted == label).sum().item()

    #             del image
    #             del label

    #     model_accuracy = correct / total
    #     print("Determined accuracy for ", self.model_package + ":", model_accuracy)
    #     self.model.to('cpu')

    #     return model_accuracy
    def evaluate(self):
        os.listdir(self.model_package)
        code = read_py_file_as_string(self.model_package+'/code.py')
        res = api.check_nn(code, self.task, self.dataset, self.metric, self.prm, self.save_to_db, self.prefix, self.save_path)
        return res
        
    def get_args(self):
        args = {
            'model_package': self.model_package,
            'task': self.task,
            'dataset': self.dataset,
            'metric': self.metric,
            'prm': self.prm,
        }
        return args
