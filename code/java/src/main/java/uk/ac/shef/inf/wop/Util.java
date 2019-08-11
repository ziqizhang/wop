package uk.ac.shef.inf.wop;

import org.apache.commons.lang.StringEscapeUtils;
import org.jsoup.Jsoup;

import java.io.*;
import java.util.Arrays;
import java.util.List;

public class Util {

    private static long tooMany=0;
    private static long tooFew=0;

    private static String toASCII(String in) {
        String fold = in.replaceAll("[^\\p{ASCII}]", "").
                replaceAll("\\s+", " ").trim();
        return fold;
    }

    private static String cleanDesc(String value) {
        value = Jsoup.parse(value).text();
        try {
            value = StringEscapeUtils.unescapeJava(value);
        }catch (Exception e){
            System.out.println("\t"+value);
        }
        String asciiValue = toASCII(value);

        String alphanumeric = asciiValue.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]", " ").
                replaceAll("\\s+", " ").trim();
        //value= StringUtils.stripAccents(value);

        List<String> normTokens = Arrays.asList(alphanumeric.split("\\s+"));
        if (normTokens.size() > 1000) {
            tooMany++;
            System.out.println("> "+normTokens.size());
            return null;
        }
        if (normTokens.size() <5) {
            tooFew++;
            return null;
        }
        return asciiValue;
    }
    private static void filterDescriptions(String inFile, String outFile){

        try (BufferedReader br = new BufferedReader(new FileReader(inFile))) {
            PrintWriter p = new PrintWriter(outFile);

            long count=0;
            String line;
            while ((line = br.readLine()) != null) {
                line=cleanDesc(line);
                if (line==null)
                    continue;

                count++;
                p.println(line);
                if (count%100000==0)
                    System.out.println(count);
            }

            p.close();
            System.out.println(count);
            System.out.println("too many="+tooMany);
            System.out.println("too few="+tooFew);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args){
        filterDescriptions(args[0],args[1]);
    }
}
