// print hello world
public class hello {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}

// setup connection to database
Connection conn = DriverManager.getConnection("jdbc:postgresql://localhost:5432/postgres")
Statement stmt = conn.createStatement();

// execute query
ResultSet rs = stmt.executeQuery("SELECT * FROM table");

// print results
while (rs.next()) {
    System.out.println(rs.getString("column"));
}

// function to compute value of two numbers based on an operator  Commentary: this was typed by hand
public static int compute(int a, int b, String operator) {
    if (operator.equals("+")) {
        return a + b;
    } else if (operator.equals("-")) {
        return a - b;
    } else if (operator.equals("*")) {
        return a * b;
    } else if (operator.equals("/")) {
        return a / b;
    } else {
        return 0;
    }
}

// junit test case for compute function  Commentary:  this was typed by hand
@Test
public void testCompute() {
    assertEquals(3, compute(1, 2, "+"));
    assertEquals(-1, compute(1, 2, "-"));
    assertEquals(2, compute(1, 2, "*"));
    assertEquals(0, compute(1, 2, "/"));
    assertEquals(0, compute(1, 2, "a"));
}

