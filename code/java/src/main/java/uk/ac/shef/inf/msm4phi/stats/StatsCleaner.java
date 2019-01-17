package uk.ac.shef.inf.msm4phi.stats;

import com.opencsv.*;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * reads the three stats (content, interaction, user), then clean them as:
 *
 * 1. find tags with tweets<50
 * 2. find tags with users=0
 * 3. remove all the above tags
 * 4. remove columns from content: tweets, % replied, avg_reply_freq, %_interacted
 * 5. remove columns from users: users, %outliers+ (all such columns)
 *
 * with an option to then normalise all values by max value.
 * Also saves removed tags and their stats
 */
public class StatsCleaner {

    private static final int CONTENT_MIN_TWEETS=50;
    private static final int USER_MIN_U=10;

    private static final int CONTENT_COL_TWEET=1;
    private static final int USER_COL_U=1;

    private static final int[] REMOVE_COLS_CONTENT={0,1,2,3,4,5,6,7,13,16};
    private static final int[] REMOVE_COLS_INTERACTION={0,1,6,7,8,9};
    private static final int[] REMOVE_COLS_USER={0,1,5,6,9,10,13,14,17,18};

    public static void main(String[] args) throws IOException {
        StatsCleaner sc = new StatsCleaner();
        sc.process(args[0],args[1]);
    }

    private void process(String inFolder, String outFolder) throws IOException {
        File[] files = new File(inFolder).listFiles();
        List<String[]> contents=new ArrayList<>(), users=new ArrayList<>(), interactions=new ArrayList<>();
        sortByTag(contents);
        sortByTag(users);
        sortByTag(interactions);
        for (File f : files){
            if (f.getName().contains("content"))
                contents= Util.readCSV(f.toString());
            else if (f.getName().contains("user"))
                users= Util.readCSV(f.toString());
            else if (f.getName().contains("inter"))
                interactions= Util.readCSV(f.toString());
        }

        Set<String> removeTags=findContentToRemove(contents);
        removeTags.addAll(findUserToRemove(users));

        removeData(removeTags, contents, outFolder+"/content_cleaned.csv", REMOVE_COLS_CONTENT);
        removeData(removeTags, interactions, outFolder+"/interaction_cleaned.csv", REMOVE_COLS_INTERACTION);
        removeData(removeTags, users, outFolder+"/user_cleaned.csv", REMOVE_COLS_USER);
    }


    private void removeData(Set<String> removeTags, List<String[]> contents, String outFile,
                            int[] removeCols) throws IOException {
        Writer writer = Files.newBufferedWriter(Paths.get(outFile));
        Writer removedWriter = Files.newBufferedWriter(Paths.get(outFile+".rm.csv"));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);
        CSVWriter csvRemovedWriter = new CSVWriter(removedWriter,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        Iterator<String[]> it = contents.iterator();
        while(it.hasNext()){
            String[] values = it.next();
            if (removeTags.contains(values[0])) {
                csvRemovedWriter.writeNext(values);
                continue;
            }

            List<String> list = new ArrayList<>();
            for (int i=0; i<values.length; i++) {
                boolean skip=false;
                for (int ri : removeCols) {
                    if (i == ri) {
                        skip = true;
                        break;
                    }
                }
                if (!skip)
                    list.add(values[i]);
            }
            csvWriter.writeNext(list.toArray(new String[0]));
        }

        csvWriter.close();
        csvRemovedWriter.close();
    }

    private Set<String> findContentToRemove(List<String[]> contents) {
        Set<String> remove = new HashSet<>();
        int count=0;
        for (String[] record: contents){
            if (count==0) {
                count++;
                continue;
            }

            int c = Integer.valueOf(record[CONTENT_COL_TWEET]);
            if (c<CONTENT_MIN_TWEETS)
                remove.add(record[0]);
        }
        return remove;
    }

    private Set<String> findUserToRemove(List<String[]> users) {
        Set<String> remove = new HashSet<>();
        int count=0;
        for (String[] record: users){
            if (count==0) {
                count++;
                continue;
            }

            int c = Integer.valueOf(record[USER_COL_U]);
            if (c<USER_MIN_U)
                remove.add(record[0]);
        }
        return remove;
    }

    private void sortByTag(List<String[]> values){
        Collections.sort(values, Comparator.comparing(strings -> strings[0]));
    }

}
