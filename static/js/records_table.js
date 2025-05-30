$(document).ready(function () {
  function fetchRecords() {
    $.ajax({
      url: "/api/records/",
      method: "GET",
      success: function (data) {
        // Extract the last 7 records from the data array
        const lastSevenRecords = data.slice(-7);

        const tbody = $("#myTable tbody");
        tbody.empty();


        lastSevenRecords.forEach(function (record, index) {
          const row = $("<tr>");
          // Display the index + 1 to show the serial number starting from 1
          row.append($("<td>").text(index + 1));
          row.append($("<td>").text(record.licenseplate_no));
          row.append($("<td>").text(record.speed));
          row.append($("<td>").text(record.date));

          tbody.append(row);
        });

        //   data.forEach(function (record, index) {
        //     const row = $("<tr>");
        //     row.append($("<td>").text(index + 1));
        //     row.append($("<td>").text(record.licenseplate_no));
        //     row.append($("<td>").text(record.speed));
        //     row.append($("<td>").text(record.date));
        //     row.append($("<td>").text(record.count));
        //     tbody.append(row);
        //   });
      },
      error: function (error) {
        console.log("Error fetching records:", error);
      },
    });
  }

  // Initial fetch when the page loads
  fetchRecords();

  // Fetch records periodically (e.g., every 2 seconds) for real-time updates
  setInterval(fetchRecords, 2000); // Update every 2 seconds
});