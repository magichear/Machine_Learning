# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 22:48:56 2021

@author: L
"""
import math
class AdvCalculator(object):
    def __init__(self):
        self.stack_op=[]
        self.stack_num=[]
        self.num_str=set("0123456789.e")
        self.op_priority={    '+': {'+':-1, '-':-1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1, '!':-1, '^':-1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                              '-': {'+':-1, '-':-1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1, '!':-1, '^':-1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                              '*': {'+': 1, '-': 1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1, '!':-1, '^':-1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                              '/': {'+': 1, '-': 1, '*':-1, '/':-1, '√':-1, '(':1, ')':-1, '!':-1, '^':-1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                              '√': {'+': 1, '-': 1, '*': 1, '/': 1, '√':-1, '(':1, ')':-1, '!':-1, '^':-1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                              '(': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')': 0, '!':-1, '^': 1, 'ln': 1, 'log': 1, 'sin': 1, 'cos': 1, 'tan': 1, 'arcsin': 1, 'arccos': 1, 'arctan': 1},
                              ')': {'+':-1, '-':-1, '*':-1, '/':-1, '√':-1, '(':0, ')':-1, '!':-1, '^':-1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                              '!': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln': 1, 'log': 1, 'sin': 1, 'cos': 1, 'tan': 1, 'arcsin': 1, 'arccos': 1, 'arctan': 1},
                              '^': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!': 1, '^':-1, 'ln':-1, 'log':-1, 'sin': 1, 'cos': 1, 'tan': 1, 'arcsin': 1, 'arccos': 1, 'arctan': 1},
                             'ln': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1}, 
                            'log': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1}, 
                            'sin': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1}, 
                            'cos': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},                   
                            'tan': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},
                         'arcsin': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},  
                         'arccos': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},  
                         'arctan': {'+': 1, '-': 1, '*': 1, '/': 1, '√': 1, '(':1, ')':-1, '!':-1, '^': 1, 'ln':-1, 'log':-1, 'sin':-1, 'cos':-1, 'tan':-1, 'arcsin':-1, 'arccos':-1, 'arctan':-1},  
}
        self.op_str=set(self.op_priority.keys())
        self.op_element={'+':2,'-':2,'*':2,'/':2,'√':1,'(':0,')':0,'!':1, '^':2, 'ln':1, 'log':1, 'sin':1, 'cos':1, 'tan':1, 'arcsin':1, 'arccos':1, 'arctan':1}
    
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
            elif op=='^':self.stack_num.append(b**a)
        elif self.op_element[op]==1:
            a=self.stack_num.pop()
            if op=='√':
                # print(op+str(a))
                self.stack_num.append(a**0.5)
            elif op=='!':
                ans=1
                while a:
                    ans*=(a)
                    a-=1
                self.stack_num.append(ans)
            elif op=='sin':
                #注意a是角度，需要转成弧度
                self.stack_num.append(math.sin(a/180*math.pi))
            elif op=='cos':
                self.stack_num.append(math.cos(a/180*math.pi))
            elif op=='tan':
                self.stack_num.append(math.tan(a/180*math.pi))
            elif op=='ln':
                self.stack_num.append(math.log(a))
            elif op=='log':
                self.stack_num.append(math.log2(a))
            elif op=='arcsin':
                self.stack_num.append(math.asin(a)/math.pi*180)
            elif op=='arccos':
                self.stack_num.append(math.acos(a)/math.pi*180)
            elif op=='arctan':
                self.stack_num.append(math.atan(a)/math.pi*180)
    def solver(self, input):
        self.stack_op.clear()
        self.stack_num.clear()
        i=0
        num=''
        last_char=''
        for i in range(len(input)):
            #注意处理大于9的数字，1 4
            # print(self.stack_num,self.stack_op)
            char=input[i]
            if char in self.num_str:
                num+=char
            else:
                if num !='':
                    if num=='e':num=math.e
                    self.stack_num.append(float(num))
                    num=''
                    last_char=''
                last_char+=char
                if last_char not in self.op_str:
                    continue
                # elif char=='-' and last_char!=')':
                    
                    
                #判断运算符的优先级
                # print('---',last_char,self.stack_op)
                # op=char
                op=''
                while self.stack_op and self.op_priority[last_char][self.stack_op[-1]]<=0:#优先级相同或更小
                    if self.op_priority[last_char][self.stack_op[-1]]<0:
                        #弹出符号，先进行处理,处理完再压栈
                        self.run_operator()
                    else:
                        op=self.stack_op.pop()
                        break
                else:
                    if last_char=='-':
                        #需要加0的情况
                        #(-9)
                        #-9
                        if (not self.stack_num) or (input[i-1] in self.op_str and input[i-1]!=')' and input[i-1]!='!'):
                        #if not self.stack_num and (not self.stack_op or self.stack_op[-1]=='(' ):
                            self.stack_num.append(0)
                    self.stack_op.append(last_char)
                last_char=''
            #last_char=char
        if num!='':
            if num=='e':num=math.e
            self.stack_num.append(float(num))
        # print(self.stack_num,self.stack_op)
        while self.stack_op:
            self.run_operator()
        return self.stack_num[0] 


def test_Calculator():
    calculator = AdvCalculator()
    input = "(2+3)*4"
    ans = calculator.solver(input=input)
    # (2+3)*4=20.0
    print("%s=%s" % (input, ans))

    input = "(2+3)*4+√9*(1/2+((1+1)*2))"
    ans = calculator.solver(input=input)
    # (2+3)*4+√9*(1/2+((1+1)*2))=33.5
    print("%s=%s" % (input, ans))

    input = "(2+3)*4+√9"
    ans = calculator.solver(input=input)
    # (2+3)*4+√9=23.0
    print("%s=%s" % (input, ans))

    input = "sin(2^6+26)"
    ans = calculator.solver(input=input)
    # sin(2^6+26)=1.0
    print("%s=%s" % (input, ans))

    input = "2^6"
    ans = calculator.solver(input=input)
    # 2^6=64.0
    print("%s=%s" % (input, ans))

    input = "2+arcsin(log(2.0))+5+2^(2)+(-log(2))"
    ans = calculator.solver(input=input)
    # 2+arcsin(log(2.0))+5+2^(2)+(-log(2))=100.0
    print("%s=%s" % (input, ans))

    input = "2^log(8)"
    ans = calculator.solver(input=input)
    # 2^log(8)=8.0
    print("%s=%s" % (input, ans))

    input = "2^(-1)"
    ans = calculator.solver(input=input)
    # 2^(-1)=0.5
    print("%s=%s" % (input, ans))

    input = "-2*3*2^(-1)+2^2^3+3!"
    ans = calculator.solver(input=input)
    # -2*3*2^(-1)+2^2^3+3!=67.0
    print("%s=%s" % (input, ans))

    input = "8^1-5!+10"
    ans = calculator.solver(input=input)
    # 8^1-5!+10=-102.0
    print("%s=%s" % (input, ans))

    input = "(5!-8^(1/2))*e"
    ans = calculator.solver(input=input)
    # (5!-8^(1/2))*e=318.5053573587672
    print("%s=%s" % (input, ans))

    input = "(2-1*2+14/2)+√9"
    ans = calculator.solver(input=input)
    # (2-1*2+14/2)+√9=10.0
    print("%s=%s" % (input, ans))

    input = "4^(-1/2)"
    ans = calculator.solver(input=input)
    # 4^(-1/2)=0.5
    print("%s=%s" % (input, ans))

    input = "ln(e*e)"
    ans = calculator.solver(input=input)
    # ln(e*e)=2.0
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