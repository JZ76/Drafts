import edu.princeton.cs.algs4.In;

import java.time.LocalDate;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

public class test {
    public static void main(String[] args) throws InterruptedException {
        int[] aaa = new int[]{3,5,7};
        saaa a = new saaa();
        int[] nums1 = new int[]{1,2,3,2,1,2};
        int[] nums2 = new int[]{10,5,2,1,5,2};
        int[][] c1 = new int[][]{
                {9,1,7,7,8,4,5,10,4,5,3,9,10,9,5,2},
                {2,3,1,8,7,1,10,9,4,6,2,1,9,7,7,2},
                {7,3,9,9,9,8,8,10,7,10,1,10,1,7,10,4},
                {10,4,2,3,6,5,9,7,7,5,5,8,5,1,5,2},
                {2,9,7,6,3,7,9,1,8,2,6,10,3,3,6,8},
                {1,4,5,6,4,8,1,7,7,5,2,2,4,4,8,9},
                {10,7,10,9,7,4,4,4,6,9,7,6,6,10,7,10},
                {10,4,8,8,6,6,9,3,9,6,4,6,1,7,10,1},
                {4,9,5,7,9,3,9,3,10,6,2,10,7,1,6,9},
                {3,4,1,10,7,2,9,1,3,2,4,6,8,3,2,6}
        };
        int[][] c2 = new int[][]{{1,1,1,0,0}, {0,0,1,1,1},{0,1,0,0,0}, {1,0,1,1,0}, {0,1,0,1,0}};



        String[] y = new String[]{"caaaaa","aaaaaaaaa","a","bbb","bbbbbbbbb","bbb","cc","cccccccccccc","ccccccc","ccccccc","cc","cccc","c","cccccccc","c"};
        String[] yy = new String[]{"ab", "ab"};
        System.out.println(Arrays.deepToString(a.rotateGrid(c1, 5)));

        //System.out.println(a);
        //System.out.println(a.isValid("{[()()]}"));
        //System.out.println(a.change(10,new int[]{1,2,5}));
        //System.out.println(a.canPartition(new int[]{1,5,11,5}));
        //System.out.println(a.findTargetSumWays(nums2,3));
        //System.out.println(a.wordBreak("catsandog", Arrays.asList(y.clone())));
        //Pattern p = Pattern.compile("v*a*b");
        //Matcher m = p.matcher(s1);
        //System.out.println(m.matches());
    }
}
