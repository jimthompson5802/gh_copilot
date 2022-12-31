/* function to add two numbers */
int add(int a, int b)
{
    return a + b;
}

/* function to add, subtract, multiply and divide two numbers based on an operator */
int calculate(int a, int b, char op)
{
    int result = 0;
    switch (op)
    {
        case '+':
            result = add(a, b);
            break;
        case '-':
            result = subtract(a, b);  /* Commentary: not sure where subtract is defined */
            break;
        case '*':
            result = multiply(a, b);  /* Commentary: not sure where multiply is defined */
            break;
        case '/':
            result = divide(a, b);  /* Commentary: not sure where divide is defined */
            break;
        default:
            printf("Invalid operator");
    }
    return result;
}

/* main function */
int main()
{
    int a, b, result;
    char op;
    printf("Enter two numbers: ");
    scanf("%d %d", &a, &b);
    printf("Enter an operator: ");
    scanf(" %c", &op);
    result = calculate(a, b, op);
    printf("Result: %d", result);
    return 0;
}

