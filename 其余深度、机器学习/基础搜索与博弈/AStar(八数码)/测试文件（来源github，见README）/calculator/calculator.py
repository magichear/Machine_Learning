# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 22:48:56 2021

@author: L
"""
class Calculator(object):
    def __init__(self):
        self.stack_op=[]
        self.stack_num=[]
        self.op_priority={  '+': {'+':-1, '-':-1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1},
                            '-': {'+':-1, '-':-1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1},
                            '*': {'+': 1, '-': 1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1},
                            '/': {'+': 1, '-': 1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1},
                            '√': {'+': 1, '-': 1, '*': 1, '/': 1, '√':-1, '(':1, ')':-1},
                            '(': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')': 0},
                            ')': {'+':-1, '-':-1, '*':-1, '/':-1, '√':-1, '(':0, ')':-1},}
        self.op_element={'+':2,'-':2,'*':2,'/':2,'√':1,'(':0,')':0}
    
    # 系统会默认调用这个函数进行评测，你必须实验这个函数
    # 输入一个计算表达式，例如2+3*√4，返回的结果是8
    # 返回计算后的结构
    def run_operator(self):
        op=self.stack_op.pop()
        if self.op_element[op]==2:#二元运算
            a=self.stack_num.pop()
            b=self.stack_num.pop()
            # print(str(b)+op+str(a))
            if op=='*':self.stack_num.append(b*a)
            elif op=='+':self.stack_num.append(b+a)
            elif op=='-':self.stack_num.append(b-a)
            elif op=='/':self.stack_num.append(b/a)
        elif self.op_element[op]==1:
            a=self.stack_num.pop()
            if op=='√':
                # print(op+str(a))
                self.stack_num.append(a**0.5)
    def solver(self, input):
        self.stack_op.clear()
        self.stack_num.clear()
        i=0
        num=0
        last_char=''
        for i in range(len(input)):
            #注意处理大于9的数字，1 4
            # print(self.stack_num,self.stack_op)
            char=input[i]
            if char.isdigit():
                num=10*num+(int(char))
            else:
                if str(last_char).isdigit():
                    self.stack_num.append(num)
                    num=0
                elif char=='-' and last_char!=')':
                    self.stack_num.append(0)
                #判断运算符的优先级
                # print('---',char,self.stack_op)
                # op=char
                while self.stack_op and self.op_priority[char][self.stack_op[-1]]<=0:#优先级相同或更小
                    if self.op_priority[char][self.stack_op[-1]]<0:
                        #弹出符号，先进行处理,处理完再压栈
                        self.run_operator()
                    else:
                        op=self.stack_op.pop()
                        break
                else:
                    self.stack_op.append(char)
            last_char=char
        if str(last_char).isdigit():
            self.stack_num.append(num)
        # print(self.stack_num,self.stack_op)
        while self.stack_op:
            self.run_operator()
        return self.stack_num[0] 


def test_Calculator():
    
    calculator = Calculator()
    # input = "2+3*4"
    # ans = calculator.solver(input=input)
    # # 2+3*4=14
    # print("%s=%s" % (input, ans)) 

    # input = "(2+3)*4"
    # ans=calculator.solver(input=input)
    # # (2+3)*4=20
    # print("%s=%s" % (input, ans))

    input = "(2+3)*4+√9*(1/2+((1+1)*2))"
    ans = calculator.solver(input=input)
    # (2+3)*4+√9*(1/2+((1+1)*2))=33.5
    print("%s=%s" % (input, ans))
    
    input = "-2+(-3)*5"
    ans = calculator.solver(input=input)
    # -2+(-3)*5=-17
    print("%s=%s" % (input, ans))

    input = "(2-1*2+14/2)+√9"
    ans = calculator.solver(input=input)
    # (2-1*2+14/2)+√9=10.0
    print("%s=%s" % (input, ans))
    
    
    input = "(1+(4+5+2)-3)+(6+8)"
    ans = calculator.solver(input=input)
    # (1+(4+5+2)-3)+(6+8)=23
    print("%s=%s" % (input, ans))
    input = "(7)-(0)+(4)"
    ans = calculator.solver(input=input)
    # (7)-(0)+(4)=11
    print("%s=%s" % (input, ans))
    
if __name__=="__main__":
    test_Calculator()