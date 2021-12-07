package uk.ac.shef.inf.wop.goldstandard;

import com.opencsv.CSVWriter;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * takes the GS1 classification data folder, process each lvl1 class and create a CSV containing the
 * classification taxonomy in the following format:
 *
 * lvl3 class, lvl2 class, lvl1 class
 * lvl3 class, lvl2 class, lvl1 class
 * etc.
 *
 */
public class GS1ClassExtractor {

    public static void main(String[] args) throws IOException, InvalidFormatException {

        process("/home/zz/Work/data/wdc/GS1_en_2019-06/EN",
                "/home/zz/Work/data/wdc/GS1_en_2019-06/GS1_en_2019-06.csv",
                0,2,4);
    }


    public static void process(String inExcelFileFolder, String outCSV,
                               int lvl1Col, int lvl2Col, int lvl3Col) throws IOException, InvalidFormatException {
        FileWriter outputfile = new FileWriter(outCSV);
        // create CSVWriter object filewriter object as parameter
        CSVWriter writer = new CSVWriter(outputfile);

        // adding header to csv
        String[] header = { "Lvl3", "Lvl2", "Lvl1" };
        writer.writeNext(header);

        Set<String> lines=new HashSet<>();
        for (File folder: new File(inExcelFileFolder).listFiles()){
            if (!folder.isDirectory())
                continue;

            File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".xlsx"));

            if(files==null || files.length!=1) {
                System.err.println("Error, files not expected:" + folder);
                continue;
            }
            File spreadsheet=files[0];
            System.out.println(spreadsheet);

            Workbook workbook = WorkbookFactory.create(spreadsheet);
            Sheet sheet = workbook.getSheet("Attribute Classification Sheet");
            if (sheet==null){
                System.err.println("\t expected sheet not found for: "+spreadsheet);
                continue;
            }
            Iterator<Row> rowIterator = sheet.rowIterator();
            int rows=0;
            DataFormatter dataFormatter = new DataFormatter();
            while (rowIterator.hasNext()) {
                Row row = rowIterator.next();
                if (rows==0) {
                    rows++;
                    continue;
                }

                String lvl1code=dataFormatter.formatCellValue(row.getCell(lvl1Col));
                String lvl1text=dataFormatter.formatCellValue(row.getCell(lvl1Col+1));
                String lvl2code=dataFormatter.formatCellValue(row.getCell(lvl2Col));
                String lvl2text=dataFormatter.formatCellValue(row.getCell(lvl2Col+1));
                String lvl3code=dataFormatter.formatCellValue(row.getCell(lvl3Col));
                String lvl3text=dataFormatter.formatCellValue(row.getCell(lvl3Col+1));

                StringBuilder sb = new StringBuilder(lvl1code);
                sb.append("_").append(lvl1text).append("|")
                        .append(lvl2code).append("_").append(lvl2text).append("|")
                        .append(lvl3code).append("_").append(lvl3text).append("|");
                lines.add(sb.toString());
            }
        }

        List<String> linesSorted = new ArrayList<>(lines);
        Collections.sort(linesSorted);
        for (String l: linesSorted){
            String[] values = l.split("\\|");
            writer.writeNext(values);
        }

        writer.close();
    }
}
