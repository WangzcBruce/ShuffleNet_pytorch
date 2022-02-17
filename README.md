# ShuffleNet_pytorch
The implementation of ShuffleNet based on the Pytorch.
# 
The pytorch code comes from the blog of "https://zhuanlan.zhihu.com/p/32304419", thanks the author a lot.
I read the code and give the detailed description of each step and find out the principle of group convolution in the try code.  
For example, input the data --> 1 * 24 * 32 * 32   using the conv2d (input_channels=24, output_channels=12, ...., groups=3)  
the input/output channels must be n times of the group. The convert matrix above is (12,8,32,32)^T , each group performs once multiply ,then we add the groups' result to obtain the final GConv result.  
