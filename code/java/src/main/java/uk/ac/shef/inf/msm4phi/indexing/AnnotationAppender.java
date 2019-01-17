package uk.ac.shef.inf.msm4phi.indexing;

import com.opencsv.CSVReader;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * process annotated files (example by default are in:/home/zz/Work/msm4phi_data/paper2/output)
 * and add the annotated label to the solr index
 */
public class AnnotationAppender {
    public void process(String folder, int idCol, int labelCol, SolrClient client) throws IOException, SolrServerException {

        int count = 0;
        for (File f : new File(folder).listFiles()) {
            System.out.println(f);
            Reader reader = Files.newBufferedReader(Paths.get(f.toString()));
            CSVReader csvReader = new CSVReader(reader);

            // Reading Records One by One in a String array
            int lines = 0;
            String[] nextRecord;
            while ((nextRecord = csvReader.readNext()) != null) {
                lines++;
                if (lines == 1)
                    continue;
                if (nextRecord.length<18) {
                    System.err.println(String.format("line %d has missing columns", lines));
                    continue;
                }
                String id = nextRecord[idCol];
                String label = nextRecord[labelCol];

                SolrQuery q = new SolrQuery();
                q.setQuery("user_screen_name:" + id);
                q.setStart(0);
                q.setRows(1);
                q.setFields("*");

                boolean stop = false;
                while (!stop) {
                    QueryResponse res = null;
                    long total = 0;
                    try {
                        res = Util.performQuery(q, client);
                        if (res != null) {
                            total = res.getResults().getNumFound();
                            //update results
                            update(res, client, label);
                        }

                    } catch (Exception e) {
                        System.out.println(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                                q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
                    }

                    int curr = q.getStart() + q.getRows();
                    if (curr < total)
                        q.setStart(curr);
                    else
                        stop = true;

                }

                count++;
                if (count % 5000 == 0) {
                    System.out.println(count);
                    client.commit();
                }
            }
        }
        client.commit();
        client.close();
    }

    private void update(QueryResponse res, SolrClient solrClient, String label) {
        int count = 0;
        List<SolrInputDocument> updates = new ArrayList<>();
        for (SolrDocument d : res.getResults()) {
            SolrInputDocument updateDoc = new SolrInputDocument();
            updateDoc.addField("id", d.getFieldValue("id"));
            Map<String, Object> fieldFavModifier = new HashMap<>(1);
            fieldFavModifier.put("set", label);
            updateDoc.addField("label_s", fieldFavModifier);

            updates.add(updateDoc);
        }

        try {
            solrClient.add(updates);

        } catch (Exception e) {
            System.err.println(String.format("\t\tfailed to commit to solr with exception: \n\t %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }
}

