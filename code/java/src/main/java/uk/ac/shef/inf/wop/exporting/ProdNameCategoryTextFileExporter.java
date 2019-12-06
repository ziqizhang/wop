package uk.ac.shef.inf.wop.exporting;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.FacetField;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.wop.Util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/*
 * This class reads the index created by ProdNameCategoryExporter, to export product name and cat label into batches of
 * txt files to be used for training MT
 *
 * - also split with '-'
 *  - if name=cat, ignore
 *  - following hosts ignored:
 *      + hosts end with .ru, .rs. .gr .pl .md .cz .ee .sk .si .be .de .nl .es
 *      + www.edilportale.com
 */
public class ProdNameCategoryTextFileExporter {
    private static final Logger LOG = Logger.getLogger(ProdNameCategoryTextFileExporter.class.getName());
    private long maxLinesPerFile = 100000;
    //private long maxWordsPerFile=500;
    private int catFilecounter = 0;
    private CSVWriter catFile;
    private List<String> validDomains = Arrays.asList(".uk", ".com", ".net", ".org", ".au", ".ag",
            ".bs", ".bb", ".ca", ".do", ".gd", ".gy", ".ie", ".jm", ".nz", ".kn", ".lc", ".vc", ".tt", ".us");


    private void export(SolrClient prodcatIndex, int resultBatchSize, String outFolder) throws IOException, SolrServerException {
        List<String> largetHosts = getLargeHosts(100, 100,prodcatIndex);

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
                    int lines = exportRecord(d, catFile, largetHosts);
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

    private boolean isValidHost(String host) {
        for (String d : validDomains) {
            if (host.endsWith(d))
                return true;
        }

        return false;
    }

    private int exportRecord(SolrDocument d,
                             CSVWriter catFile, List<String> validHosts) {

        Object nameData = d.getFieldValue("name");
        Object catData = d.getFirstValue("category");
        String host = d.getFieldValue("host").toString();

        if (!validHosts.contains(host))
            return 0;

        if (isValidHost(host)) {
            //System.out.println("\tinvalid host:"+host);
            return 0;
        }

        if (nameData != null && catData != null) {
            String name = nameData.toString().trim();
            String cat = catData.toString().trim();
            if (name.equalsIgnoreCase(cat))
                return 0;

            name = cleanNameData(name);
            cat = cleanCatData(cat);

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
        if (value.contains("http://") || value.contains("https://"))
            return null;
        String asciiValue = toASCII(value);
        asciiValue = asciiValue.replaceAll("set=", "/");

        String alphanumeric = asciiValue.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}\\|\\>/]", " ").
                replaceAll("\\s+", " ").toLowerCase().trim();
        //value= StringUtils.stripAccents(value);

        int nums = Util.replacePatterns(alphanumeric, Util.numericP);
        int an = Util.replacePatterns(alphanumeric, Util.alphanumP);
        int num_or_an = nums + an;

        String alphanumeric_clean = alphanumeric.replaceAll(Util.alphanum, "LETTERNUMBER");
        alphanumeric_clean = alphanumeric_clean.replaceAll(Util.numeric, "NUMBER");


        //value= StringUtils.stripAccents(value);

        List<String> normTokens = Arrays.asList(alphanumeric_clean.split("\\s+"));
        if (normTokens.size() > 10 || normTokens.size() < 2)
            return null;
        if (num_or_an > ((double) normTokens.size() / 3.0))
            return null;
        if (normTokens.contains("http") || normTokens.contains("https") || normTokens.contains("www"))
            return null;
        if (alphanumeric.length() < 3)
            return null;

        String[] pathElements = alphanumeric_clean.split("[\\|\\>/\\-]+");
        StringBuilder sb = new StringBuilder();
        for (String path : pathElements) {
            path = path.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]", " ").
                    replaceAll("\\s+", " ").toLowerCase().trim();
            if (path.length() > 2)
                sb.append(path.replaceAll("\\s+", "_")).append(" ");
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

        int nums = Util.replacePatterns(alphanumeric, Util.numericP);
        int an = Util.replacePatterns(alphanumeric, Util.alphanumP);
        int num_or_an = nums + an;

        String alphanumeric_clean = alphanumeric.replaceAll(Util.alphanum, "LETTERNUMBER");
        alphanumeric_clean = alphanumeric_clean.replaceAll(Util.numeric, "NUMBER");

        List<String> tokens = Arrays.asList(alphanumeric_clean.split("\\s+"));
        if (tokens.size() > 10 || tokens.size() < 2)
            return null;
        if (num_or_an > ((double) tokens.size() / 3.0))
            return null;
        if (tokens.contains("http") || tokens.contains("https") || tokens.contains("www"))
            return null;

        return alphanumeric_clean;
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

    private List<String> getLargeHosts(int minResults, int topN, SolrClient prodcatIndex) throws IOException, SolrServerException {
        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setFacet(true);
        query.setFacetLimit(-1);
        query.setFacetMinCount(minResults);
        query.addFacetField("host");

        QueryResponse qr = prodcatIndex.query(query);
        FacetField ff = qr.getFacetField("host");
        List<String> hosts = new ArrayList<>();
        Map<String, Long> freq=new HashMap<>();

        for (FacetField.Count c : ff.getValues()) {
            if (isValidHost(c.getName()))
                continue;
            freq.put(c.getName(), c.getCount());
            hosts.add(c.getName());
        }

        hosts.sort((s, t1) -> freq.get(t1).compareTo(freq.get(s)));

        List<String> selected=new ArrayList<>();
        for (int i=0;i<topN && i<hosts.size();i++)
            selected.add(hosts.get(i));

        return selected;
    }

    private static void convert(String inFolder, String outFolder) throws IOException {
        long total = 0;
        for (File f : new File(inFolder).listFiles()) {
            Reader reader = Files.newBufferedReader(Paths.get(f.toString()));
            CSVReader csvReader = new CSVReader(reader);
            List<String[]> all = csvReader.readAll();
            total += all.size();

            String outFile = outFolder + "/" + f.getName();
            FileWriter outputfile = new FileWriter(outFile, false);
            // create CSVWriter object filewriter object as parameter
            CSVWriter csvWriter = new CSVWriter(outputfile, ',', '"');
            for (String[] line : all) {
                if (line.length < 2)
                    continue;
                String[] values = new String[2];
                values[0] = line[0];

                values[1] = line[1].replaceAll("_", " ").trim();
                csvWriter.writeNext(values);
            }
            csvReader.close();
            csvWriter.close();
        }
        System.out.println(total);
    }

    public static void main(String[] args) throws IOException, SolrServerException {
        convert("/home/zz/Work/data/wop_data/mt/product/mt_corpus/cat_labels",
                "/home/zz/Work/data/wop_data/mt/product/mt_corpus/cat_label_words");
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
