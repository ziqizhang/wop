package uk.ac.shef.inf.wop.exporting;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/*
 * This class reads the index created by ProdNameCategoryExporter, to export product name and cat label into batches of txt files
 */
public class ProdNameCategoryTextFileExporter {
    private static final Logger LOG = Logger.getLogger(ProdNameCategoryTextFileExporter.class.getName());
    private long maxLinesPerFile = 100000;
    //private long maxWordsPerFile=500;
    private int catFilecounter = 0;
    private CSVWriter catFile;

    private void export(SolrClient prodcatIndex, int resultBatchSize, String outFolder) throws IOException {
        int start = 0;
        SolrQuery q = createQuery(resultBatchSize, start);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        FileWriter outputfile = new FileWriter(outFolder + "/c_" + catFilecounter + ".csv", true);
        // create CSVWriter object filewriter object as parameter
        catFile = new CSVWriter(outputfile, ',', '"');

        int countCatFileLines = 0;

        while (!stop) {
            try {
                res = prodcatIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));

                for (SolrDocument d : res.getResults()) {
                    //process and export to the other solr index
                    int lines = exportRecord(d, catFile);
                    countCatFileLines += lines;
                }

                if (countCatFileLines >= maxLinesPerFile) {
                    catFile.close();
                    catFilecounter++;
                    catFile = new CSVWriter(

                            new FileWriter(outFolder + "/c_" + catFilecounter + ".csv", true));
                    countCatFileLines = 0;
                }


                start = start + resultBatchSize;
            } catch (Exception e) {
                LOG.warn(String.format("\t\t unable to successfully index product triples starting from index %s. Due to error: %s",
                        start,
                        ExceptionUtils.getFullStackTrace(e)));

            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;
        }

        try {
            prodcatIndex.close();
        } catch (Exception e) {
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    private int exportRecord(SolrDocument d,
                             CSVWriter catFile) {

        Object nameData = d.getFieldValue("name");
        Object catData = d.getFirstValue("category");

        if (nameData != null && catData != null) {
            String name = cleanNameData(nameData.toString());
            String cat = cleanCatData(catData.toString());

            if (name == null || cat == null) {
                //System.out.println("deleted: " + nameData + " |||| " + catData);
                return 0;
            }

            String[] values = {name, cat};

            catFile.writeNext(values);
            return 1;
        }

        return 0;
    }

    private String cleanCatData(String value) {
        value = StringEscapeUtils.unescapeJava(value);
        String asciiValue = toASCII(value);

        String alphanumeric = asciiValue.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]", " ").
                replaceAll("\\s+", " ").toLowerCase().trim();
        //value= StringUtils.stripAccents(value);

        List<String> normTokens = Arrays.asList(alphanumeric.split("\\s+"));
        if (normTokens.size() > 20)
            return null;
        if (normTokens.contains("http") || normTokens.contains("https") || normTokens.contains("www"))
            return null;
        if (alphanumeric.length() < 3)
            return null;

        String[] pathElements=asciiValue.split("[\\|\\>/]+");
        StringBuilder sb = new StringBuilder();
        for (String path: pathElements){
            path=path.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]", " ").
                    replaceAll("\\s+", " ").toLowerCase().trim();
            if (path.length()>2)
                sb.append(path.replaceAll("\\s+","_")).append(" ");
        }
        return sb.toString().trim();
        //return alphanumeric.trim();
    }

    private String cleanNameData(String value) {
        value = StringEscapeUtils.unescapeJava(value);
        String asciiValue = toASCII(value);
        String alphanumeric = asciiValue.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]", " ").
                replaceAll("\\s+", " ").trim().toLowerCase();
        //value= StringUtils.stripAccents(value);

        List<String> tokens = Arrays.asList(alphanumeric.split("\\s+"));
        if (asciiValue.split("\\s+").length > 30)
            return null;
        if (tokens.contains("http") || tokens.contains("https") || tokens.contains("www"))
            return null;
        if (alphanumeric.length() < 3)
            return null;
        return alphanumeric;
    }

    private static String toASCII(String in) {
        String fold = in.replaceAll("[^\\p{ASCII}]", "").replaceAll("\\s+", " ").trim();
        return fold;
    }

    private SolrQuery createQuery(int resultBatchSize, int start) {
        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setSort("random_1234", SolrQuery.ORDER.desc);
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    private static void convert(String inFolder, String outFolder) throws IOException {
        long total=0;
        for (File f : new File(inFolder).listFiles()){
            Reader reader = Files.newBufferedReader(Paths.get(f.toString()));
            CSVReader csvReader = new CSVReader(reader);
            List<String[]> all = csvReader.readAll();
            total+=all.size();

            String outFile = outFolder+"/"+f.getName();
            FileWriter outputfile = new FileWriter(outFile, false);
            // create CSVWriter object filewriter object as parameter
            CSVWriter csvWriter = new CSVWriter(outputfile, ',', '"');
            for (String[] line : all){
                String[] values = new String[2];
                values[0]=line[0];
                values[1]=line[1].replaceAll("_"," ").trim();
                csvWriter.writeNext(values);
            }
            csvReader.close();
            csvWriter.close();
        }
        System.out.println(total);
    }

    public static void main(String[] args) throws IOException {
        convert("/home/zz/Work/data/wdc/mt_corpus/cat_labels",
                "/home/zz/Work/data/wdc/mt_corpus/cat_label_words");
        System.exit(0);
        /*String weirdString="home >> cat and >> monkey";

        String[] parts=weirdString.split("[\\|\\>/]+");
        System.out.println(toASCII(weirdString));
        System.exit(0);*/

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodCatIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcat"));

        ProdNameCategoryTextFileExporter exporter = new ProdNameCategoryTextFileExporter();
        //exporter.export(prodTripleIndex, Integer.valueOf(args[2]), prodNameDescIndex);
        exporter.export(prodCatIndex, 50000, args[1]);
        prodCatIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }

}
