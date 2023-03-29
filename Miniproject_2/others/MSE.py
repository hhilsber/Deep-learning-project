class MSE(object):
    def forward(sample,label):
        #return (sample-label).pow(2).sum()
        return (sample-label).pow(2).mean()

    def backward(sample,label):
        #return 2*(sample-label)
        return 2*(sample-label)/sample.numel()
