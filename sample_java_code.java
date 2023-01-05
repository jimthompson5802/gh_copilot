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
