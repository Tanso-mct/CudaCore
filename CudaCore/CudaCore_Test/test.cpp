#include "pch.h"

TEST(TestCaseName, TestName) 
{
    int num = 10;
    CudaCore::SampleFunc(num);
    EXPECT_EQ(num, 1);
}