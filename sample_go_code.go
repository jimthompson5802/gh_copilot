// print hello world
package main

import "fmt"

func main() {
    fmt.Println("hello world")
}

// function to add two numbers
func add(a int, b int) int {
    return a + b
}

// read two numbers from stdin and print their sum
func main() {
    var a, b int
    fmt.Scanf("%d %d", &a, &b)
    fmt.Println(add(a, b))
}

