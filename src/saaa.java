import edu.princeton.cs.algs4.In;

import java.math.BigDecimal;
import java.text.SimpleDateFormat;
import java.util.*;


class saaa {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    class Node1 {
        public int val;
        public List<Node> neighbors;
        public Node1() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }
        public Node1(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }
        public Node1(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }

    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public int search(int[] nums, int target) {
        int lo = 0;
        int hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            }else if (nums[mid] >= nums[lo]){// 先根据 nums[mid] 与 nums[lo] 的关系判断 mid 是在左段还是右段
                // 再判断 target 是在 mid 的左边还是右边，从而调整左右边界 lo 和 hi
                if (target >= nums[lo] && target < nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            }else if (nums[mid] < nums[lo]){
                if (target > nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                }else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid]==nums[left]&&nums[mid]==nums[right]){
                left++;
                right--;
                continue;
            }
            if (nums[left]<nums[right]){
                return nums[left];
            }
            if (nums[mid] >= nums[left]) {
                left = mid + 1;
            } else if (nums[mid] < nums[left]) {
                right = mid;
            }

        }
        return nums[left];
    }

    public int searchRot(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        while (left<=right){
            int middle = left + (right - left) / 2;
            if (target==nums[middle]){
                return middle;
            }else if (nums[left]<=nums[middle]){
                if (target>=nums[left]&&target<=nums[middle]){
                    right = middle;
                }else {
                    left = middle + 1;
                }
            }else if (nums[middle]<nums[right]){
                if (target>=nums[middle]&&target<=nums[right]){
                    left = middle + 1;
                }else {
                    right = middle;
                }
            }
        }
        return -1;
    }
    public int searchRota(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            // 先根据 nums[0] 与 target 的关系判断目标值是在左半段还是右半段
            if (target >= nums[0]) {
                // 目标值在左半段时，若 mid 在右半段，则将 mid 索引的值改成 inf
                if (nums[mid] < nums[0]) {
                    nums[mid] = Integer.MAX_VALUE;
                }
            } else {
                // 目标值在右半段时，若 mid 在左半段，则将 mid 索引的值改成 -inf
                if (nums[mid] >= nums[0]) {
                    nums[mid] = Integer.MIN_VALUE;
                }
            }

            if (nums[mid] < target) {
                lo = mid + 1;
            } else if (nums[mid] > target){
                hi = mid - 1;
            }
        }
        return -1;
    }
    public boolean searchRotaDup(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        while (left<=right){
            int middle = left + (right - left) / 2;

            if (target==nums[middle]){
                return true;
            }

            if(nums[left]==nums[middle]){
                left++;
                continue;
            }

            if (nums[left]<nums[middle]){
                if (target>=nums[left]&&target<=nums[middle]){
                    right = middle;
                }else {
                    left = middle + 1;
                }
            }else if (nums[middle]<=nums[right]){
                if (target>=nums[middle]&&target<=nums[right]){
                    left = middle + 1;
                }else {
                    right = middle;
                }
            }
        }
        return false;
    }
    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid]>nums[mid + 1]){
                right = mid;
            }else if (nums[mid]<nums[mid + 1]){
                left = mid + 1;
            }
        }
        return left;
    }
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int size = arr.length;

        int left = 0;
        int right = size - k;

        while (left < right) {
            int mid = left + (right - left) / 2;
            // 尝试从长度为 k + 1 的连续子区间删除一个元素
            // 从而定位左区间端点的边界值
            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        List<Integer> res = new ArrayList<>();
        for (int i = left; i < left + k; i++) {
            res.add(arr[i]);
        }
        return res;
    }

    public char nextGreatestLetter(char[] letters, char target) {
        if (target < letters[0] || target >= letters[letters.length-1]){
            return letters[0];
        }
        int i = 0;
        int j = letters.length-1;
        while (i < j){
            int mid = i + (j - i) / 2;
            if (letters[mid] < target){
                i = mid + 1;
            }else if (letters[mid] > target){
                j = mid;
            }else if (letters[mid] == target){
                i = mid + 1;
            }
        }
        return letters[i];
    }

    public double findMedianSortedArrays(int[] A, int[] B) {
        int m = A.length;
        int n = B.length;
        if (m > n) {
            return findMedianSortedArrays(B, A); // 保证 m <= n
        }
        int iMin = 0, iMax = m;
        while (iMin <= iMax) {
            int i = (iMin + iMax) / 2;
            int j = (m + n + 1) / 2 - i;
            if (j != 0 && i != m && B[j - 1] > A[i]) { // i 需要增大
                iMin = i + 1;
            } else if (i != 0 && j != n && A[i - 1] > B[j]) { // i 需要减小
                iMax = i - 1;
            } else { // 达到要求，并且将边界条件列出来单独考虑
                int maxLeft = 0;
                if (i == 0) {
                    maxLeft = B[j - 1];
                } else if (j == 0) {
                    maxLeft = A[i - 1];
                } else {
                    maxLeft = Math.max(A[i - 1], B[j - 1]);
                }
                if ((m + n) % 2 == 1) {
                    return maxLeft;
                } // 奇数的话不需要考虑右半部分

                int minRight = 0;
                if (i == m) {
                    minRight = B[j];
                } else if (j == n) {
                    minRight = A[i];
                } else {
                    minRight = Math.min(B[j], A[i]);
                }

                return (maxLeft + minRight) / 2.0; //如果是偶数的话返回结果
            }
        }
        return 0.0;
    }
    public int findKthNumber(int n, int k) {
        int cur = 1;//第一字典序小的(就是1)
        int prefix = 1;// 前缀从1开始
        while (cur < k) {
            int tmp = count(n, prefix); //当前prefix峰的值
            if (tmp + cur > k) {// 找到了
                prefix *= 10; //往下层遍历
                cur++;//一直遍历到第K个推出循环
            } else {
                prefix++;//去下个峰头(前缀)遍历
                cur += tmp;//跨过了一个峰(前缀)
            }
        }//退出循环时 cur==k 正好找到
        return prefix;
    }

    private int count(int n, int prefix) {
        //不断向下层遍历可能一个乘10就溢出了, 所以用long
        long cur = prefix;
        long next = cur + 1;//下一个前缀峰头
        int count = 0;
        while (cur <= n) {
            count += Math.min(n + 1, next) - cur;//下一峰头减去此峰头
            // 如果说刚刚prefix是1，next是2，那么现在分别变成10和20
            // 1为前缀的子节点增加10个，十叉树增加一层, 变成了两层

            // 如果说现在prefix是10，next是20，那么现在分别变成100和200，
            // 1为前缀的子节点增加100个，十叉树又增加了一层，变成了三层
            cur *= 10;
            next *= 10; //往下层走
        }
        return count;
    }
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int len=nums.length;
        int low=0;
        int high=nums[len-1]-nums[0];

        while(low<high) {
            int mid = low + (high - low) / 2;
            int count = 0;
            int left = 0;
            for (int right = 0; right < len; right++) {
                while (nums[right] - nums[left] > mid) {
                    left++;
                }
                count += right - left;
            }

            if (count >= k) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }
    public int splitArray(int[] nums, int m) {
        if (nums == null || nums.length == 0 || m == 0) {
            return 0;
        }
        long left = 0;
        long right = 0;
        for (int num : nums) {
            left = Math.max(left, num);
            right += num;
        }
        while (left < right) {
            long mid = left + (right - left) / 2;
            int count = 1;
            int temp = 0;
            for (int n : nums) {
                temp += n;
                if (temp > mid) {
                    temp = n;
                    count++;
                }
            }
            if (count <= m) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return (int) left;
    }
    public int trap(int[] height) {
        int sum = 0;
        int[] max_left = new int[height.length];
        int[] max_right = new int[height.length];

        for (int i = 1; i < height.length - 1; i++) {
            max_left[i] = Math.max(max_left[i - 1], height[i - 1]);
        }
        for (int i = height.length - 2; i > 0; i--) {
            max_right[i] = Math.max(max_right[i + 1], height[i + 1]);
        }
        for (int i = 1; i < height.length - 1; i++) {
            int min = Math.min(max_left[i], max_right[i]);
            if (min > height[i]) {
                sum = sum + (min - height[i]);
            }
        }
        return sum;
    }
    public int longestValidParentheses(String s) {
        int n = s.length();
        int[] dp = new int[n];//dp是以i处括号结尾的有效括号长度
        int max_len = 0;
        //i从1开始，一个是单括号无效，另一个是防i - 1索引越界
        for(int i = 1; i < n; i++) {
            if(s.charAt(i) == ')') { //遇见右括号才开始判断
                if(s.charAt(i - 1) == '(') { //上一个是左括号
                    if(i < 2) { //开头处
                        dp[i] = 2;
                    } else { //非开头处
                        dp[i] = dp[i - 2] + 2;
                    }
                }
                else { //上一个也是右括号
                    if(dp[i - 1] > 0) {//上一个括号是有效括号
//pre_left为i处右括号对应左括号下标，推导：(i-1)-dp[i-1]+1 - 1
                        int pre_left = i - dp[i - 1] - 1;
                        if(pre_left >= 0 && s.charAt(pre_left) == '(') {//左括号存在且为左括号（滑稽）
                            dp[i] = dp[i - 1] + 2;
                            //左括号前还可能存在有效括号
                            if(pre_left - 1 > 0) {
                                dp[i] = dp[i] + dp[pre_left - 1];
                            }
                        }
                    }
                }
            }
            max_len = Math.max(max_len, dp[i]);
        }
        return max_len;
    }
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }

        int maxLen = 1;
        int begin = 0;

        // dp[i][j] 表示 s[i, j] 是否是回文串
        boolean[][] dp = new boolean[len][len];
        char[] charArray = s.toCharArray();

        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        for (int j = 1; j < len; j++) {
            for (int i = 0; i < j; i++) {
                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // 只要 dp[i][j] == true 成立，就表示子串 s[i..j] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }
    public boolean isMatch(String s, String p) {

        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;

        //"" 和p的匹配关系初始化，a*a*a*a*a*这种能够匹配空串，其他的是都是false。
        //  奇数位不管什么字符都是false，偶数位为* 时则: dp[0][i] = dp[0][i - 2]
        for (int i = 2; i <= n; i+= 2) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = dp[0][i - 2];
            }
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                char sc = s.charAt(i - 1);
                char pc = p.charAt(j - 1);
                if (sc == pc || pc == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (pc == '*') {
                    if (sc == p.charAt(j - 2) || p.charAt(j - 2) == '.') {
                        dp[i][j]=dp[i][j-1] || dp[i][j-2] || dp[i-1][j];
                    } else if (p.charAt(j - 2)!=sc) {
                        dp[i][j] = dp[i][j - 2];
                    }
                }
            }
        }
        return dp[m][n];
    }
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int maximumScore(int[] nums, int[] multipliers) {
        int m = multipliers.length;
        int[][]dp = new int[m + 1][m + 1];

        dp[1][0] = nums[0] * multipliers[0];
        dp[0][1] = nums[nums.length - 1] * multipliers[0];

        for (int i = 2; i <= m; i++) {
            int mul = multipliers[i - 1];
            for(int l = 0;l <= i;l++){
                int r = i - l;
                int nums_index = nums.length - (i - l);
                if(l == 0){
                    dp[l][r] = dp[l][r - 1] + mul * nums[nums_index];
                    continue;
                }
                if(r == 0){
                    dp[l][r] = dp[l - 1][r] + mul * nums[l - 1];
                    continue;
                }
                dp[l][r] = Math.max(dp[l - 1][r] + mul * nums[l - 1],dp[l][r - 1] + mul * nums[nums_index]);
            }
        }
        int ans = Integer.MIN_VALUE;
        for(int i = 0;i <= m;i++){
            ans = Math.max(dp[i][m - i],ans);
        }
        return ans;
    }
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();

        // 多开一行一列是为了保存边界条件，即字符长度为 0 的情况，这一点在字符串的动态规划问题中比较常见
        int[][] dp = new int[len1 + 1][len2 + 1];
        // 初始化：当 word2 长度为 0 时，将 word1 的全部删除即可
        for (int i = 1; i <= len1; i++) {
            dp[i][0] = i;
        }
        // 当 word1 长度为 0 时，插入所有 word2 的字符即可
        for (int j = 1; j <= len2; j++) {
            dp[0][j] = j;
        }

        // 由于 word1.charAt(i) 操作会去检查下标是否越界，因此在 Java 里，将字符串转换成字符数组是常见额操作
        char[] word1Array = word1.toCharArray();
        char[] word2Array = word2.toCharArray();
        // 递推开始，注意：填写 dp 数组的时候，由于初始化多设置了一行一列，横纵坐标有个偏移
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                // 这是最佳情况
                if (word1Array[i - 1] == word2Array[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                    continue;
                }
                // 否则在以下三种情况中选出步骤最少的，这是「动态规划」的「最优子结构」
                // 1、在下标 i 处插入一个字符
                int insert = dp[i][j - 1] + 1;
                // 2、替换一个字符
                int replace = dp[i - 1][j - 1] + 1;
                // 3、删除一个字符
                int delete = dp[i - 1][j] + 1;
                dp[i][j] = Math.min(Math.min(insert, replace), delete);

            }
        }
        return dp[len1][len2];
    }
    public int superEggDrop(int K, int N) {

        // dp[i][j]：一共有 i 层楼梯的情况下，使用 j 个鸡蛋的最少实验的次数
        // 注意：
        // 1、i 表示的是楼层的大小，不是第几层的意思，例如楼层区间 [8, 9, 10] 的大小为 3，这一点是在状态转移的过程中调整的定义
        // 2、j 表示可以使用的鸡蛋的个数，它是约束条件，我个人习惯放在后面的维度，表示消除后效性的意思

        // 0 个楼层和 0 个鸡蛋的情况都需要算上去，虽然没有实际的意义，但是作为递推的起点，被其它状态值所参考
        int[][] dp = new int[N + 1][K + 1];

        // 由于求的是最小值，因此初始化的时候赋值为一个较大的数，9999 或者 i 都可以
        for (int i = 0; i <= N; i++) {
            Arrays.fill(dp[i], i);
        }

        // 初始化：填写下标为 0、1 的行和下标为 0、1 的列
        // 第 0 行：楼层为 0 的时候，不管鸡蛋个数多少，都测试不出鸡蛋的 F 值，故全为 0
        for (int j = 0; j <= K; j++) {
            dp[0][j] = 0;
        }

        // 第 1 行：楼层为 1 的时候，0 个鸡蛋的时候，扔 0 次，1 个以及 1 个鸡蛋以上只需要扔 1 次
        dp[1][0] = 0;
        for (int j = 1; j <= K; j++) {
            dp[1][j] = 1;
        }

        // 第 0 列：鸡蛋个数为 0 的时候，不管楼层为多少，也测试不出鸡蛋的 F 值，故全为 0
        // 第 1 列：鸡蛋个数为 1 的时候，这是一种极端情况，要试出 F 值，最少次数就等于楼层高度（想想复杂度的定义）
        for (int i = 0; i <= N; i++) {
            dp[i][0] = 0;
            dp[i][1] = i;
        }

        // 从第 2 行，第 2 列开始填表
        for (int i = 2; i <= N; i++) {
            for (int j = 2; j <= K; j++) {
                for (int k = 1; k <= i; k++) {
                    // 碎了，就需要往低层继续扔：层数少 1 ，鸡蛋也少 1
                    // 不碎，就需要往高层继续扔：层数是当前层到最高层的距离差，鸡蛋数量不少
                    // 两种情况都做了一次尝试，所以加 1
                    dp[i][j] = Math.min(dp[i][j], Math.max(dp[k - 1][j - 1], dp[i - k][j]) + 1);
                }
            }
        }
        return dp[N][K];
    }
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum%2!=0){
            return false;
        }
        sum = sum/2;
        boolean[][] dp = new boolean[nums.length+1][sum+1];
        for (int i = 0; i < dp.length; i++){
            dp[i][0] = true;
        }
        for (int i = 1; i < dp[0].length; i++){
            dp[0][i] = false;
        }
        for (int i = 1; i < nums.length+1; i++) {
            for (int j = 1; j < sum + 1; j++) {
                if (j - nums[i - 1] < 0) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j]||dp[i - 1][j - nums[i - 1]];
                }
            }
        }
        return dp[nums.length][sum];
    }
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int n : nums) sum += n;
        // 这两种情况，不可能存在合法的子集划分
        if (sum < target || (sum + target) % 2 == 1) {
            return 0;
        }
        return subsets(nums, (sum + target) / 2);
    }

    private int subsets(int[] nums, int sum) {
        int n = nums.length;
        int[][] dp = new int[n + 1][sum + 1];
        // base case

        dp[0][0] = 1;


        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= sum; j++) {
                if (j >= nums[i-1]) {
                    // 两种选择的结果之和
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]];
                } else {
                    // 背包的空间不足，只能选择不装物品 i
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n][sum];
    }
    public int change(int amount, int[] coins) {
        int K = coins.length + 1;
        int I = amount + 1;
        int[][] DP = new int[K][I];

        //初始化基本状态
        for (int k = 0; k < coins.length + 1; k++){
            DP[k][0] = 1;
        }
        for (int k = 1; k <= coins.length ; k++){
            for (int i = 1; i <= amount; i++){
                if ( i >= coins[k-1]) {
                    DP[k][i] = DP[k][i-coins[k-1]] + DP[k-1][i];
                } else{
                    DP[k][i] = DP[k-1][i];
                }
            }
        }
        return DP[coins.length][amount];
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        int lengthS = s.length();
        int lengthW = wordDict.size();
        boolean[][] dp = new boolean[lengthW + 1][lengthS + 1];

        Arrays.fill(dp[0], false);

        for (int i = 0; i < lengthW + 1; i++) {
            dp[i][0] = true;
        }

        for (int i = 1; i < lengthS + 1; i++) {
            for (int j = 1; j < lengthW + 1; j++) {
                if (i - wordDict.get(j - 1).length() >= 0) {
                    String temp = s.substring(i - wordDict.get(j - 1).length(), i);
                    for (int a = 0; a < lengthW + 1; a++) {
                        if (i - wordDict.get(j - 1).length() >= 0 && dp[a][i - wordDict.get(j - 1).length()]) {
                            dp[j][i] = temp.equals(wordDict.get(j - 1));
                            break;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < lengthW + 1; i++) {
            if (dp[i][lengthS])
                return true;
        }
        return false;
    }
    public int maxProfit(int[] prices) {
        int days = prices.length;
        int[][] dp = new int[days+1][2];
        dp[0][1] = Integer.MIN_VALUE;
        dp[0][0] = 0;
        for (int i = 1; i < days+1; i++){
            dp[i][0] = Math.max(dp[i-1][1]+prices[i-1],dp[i-1][0]);
            if (i==1){
                dp[i][1] = Math.max(dp[i-1][1],0-prices[i-1]);
                continue;
            }
            dp[i][1] = Math.max(dp[i-1][1],dp[i-2][0]-prices[i-1]);
        }
        return dp[days][0];
    }
    public int[] countBits(int num) {
        String bit = Integer.toBinaryString(num);
        System.out.println(bit);
        int[] dp = new int[num + 1];
        dp[0] = 0;
        for (int i = 1; i < num + 1; i++) {
            if (i % 2 == 0) {
                dp[i] = dp[i / 2];
            } else {
                dp[i] = dp[i / 2] + 1;
            }
        }
        return dp;
    }
    public int maximalSquare(char[][] matrix) {
        int height = matrix.length;
        int width = matrix[0].length;
        int maxSide = 0;

        // 相当于已经预处理新增第一行、第一列均为0
        int[][] dp = new int[height + 1][width + 1];

        for (int row = 1; row < height; row++) {
            for (int col = 1; col < width; col++) {
                if (matrix[row - 1][col - 1] == '1') {
                    dp[row][col] = Math.min(Math.min(dp[row][col - 1], dp[row - 1][col]), dp[row - 1][col - 1]) + 1;
                    maxSide = Math.max(maxSide, dp[row][col]);
                }
            }
        }
        return maxSide * maxSide;
    }
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return len;
        }

        int[] dp = new int[len];
        Arrays.fill(dp, 1);

        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int res = 0;
        for (int i = 0; i < len; i++) {
            res = Math.max(res, dp[i]);
        }
        return res;
    }
    public void sort1(int[] arr){
        sort(arr, 0, arr.length - 1);
    }

    private static void sort(int[] a, int low, int high) {
        if (high <= low) {
            return;
        }
        int j = partition(a, low, high);
        sort(a, low, j - 1);
        sort(a, j + 1, high);
    }

    private static int partition(int[] a, int low, int high) {
        int i = low;
        int j = high + 1;
        int v = a[low];
        while (true) {
            while (a[++i]<v) {
                if (i == high) {
                    break;
                }
            }
            while (v<a[--j]) {
                if (j == low) {
                    break;
                }
            }
            if (i >= j) {
                break;
            }
            exchange(a, i, j);
        }
        exchange(a, low, j);
        return j;
    }

    private static void exchange(int[] a, int i, int j) {
        int swap = a[i];
        a[i]= a[j];
        a[j]= swap;
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null && right == null) {
            return null; // 1.
        }
        if (left == null) {
            return right; // 3.
        }
        if (right == null) {
            return left; // 4.
        }
        return root; // 2. if(left != null and right != null)
    }
    public TreeNode invertTree(TreeNode root) {
        if (root==null){
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);

        root.left = right;
        root.right = left;

        return root;
    }
    private int answer = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        if (root == null){
            return 0;
        }
        maxDepth(root);
        return answer;
    }
    public int maxDepth(TreeNode root) {
        if (root == null){
            return 0;
        }else {
            int lH = maxDepth(root.left);
            int rH = maxDepth(root.right);
            answer = Math.max(lH+rH,answer);
            return Math.max(lH,rH)+1;
        }
    }
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1==null&&root2==null){
            return null;
        }

        if (root1==null){
            return root2;
        }
        if (root2==null){
            return root1;
        }
        root1.val +=root2.val;
        TreeNode left = mergeTrees(root1.left,root2.left);
        TreeNode right = mergeTrees(root1.right,root2.right);
        root1.left = left;
        root1.right = right;
        return root1;
    }
    TreeNode pre = null;
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        flatten(root.right);
        flatten(root.left);
        root.right = pre;
        root.left = null;
        pre = root;
    }
    int sum = 0;
    public TreeNode convertBST(TreeNode root) {
        if (root==null){
            return null;
        }
        convertBST(root.right);
        root.val += sum;
        sum=root.val;
        convertBST(root.left);
        return root;
    }
    private int ans = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if (root == null){
            return 0;
        }
        maxDept(root);
        return ans;
    }
    public int maxDept(TreeNode root) {
        if (root == null){
            return 0;
        }else {
            int lH = Math.max(maxDept(root.left),0);
            int rH = Math.max(maxDept(root.right),0);
            ans = Math.max(lH+rH+root.val,ans);
            return Math.max(rH,lH)+root.val;
        }
    }
    public ListNode mergeKLists(ListNode[] lists) {
        int n = lists.length;
        if (n==1){
            return lists[0];
        }
        ListNode pre = new ListNode(-10001);
        for (int i = 1; i < n; i++){
            ListNode prev = pre;
            while (lists[0]!=null&&lists[i]!=null){
                if (lists[0].val>lists[i].val){
                    prev.next = lists[i];
                    lists[i] = lists[i].next;
                }else {
                    prev.next = lists[0];
                    lists[0] = lists[0].next;
                }
                prev = prev.next;
            }
            if (lists[0]==null){
                prev.next = lists[i];
            }else {
                prev.next = lists[0];
            }
            lists[0] = pre.next;
        }
        return pre.next;
    }
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        return threeSum1(nums,0);
    }
    public List<List<Integer>> threeSum1(int[] nums,int tgt) {
        List<List<Integer>> answer = new LinkedList<>();
        int len = nums.length;
        for (int i = 0; i < len; i++){
            List<List<Integer>> t = twoSum(nums,tgt-nums[i],i+1);
            for (List<Integer> s : t){
                s.add(nums[i]);
                answer.add(s);
            }
            while (i<len-1&&nums[i]==nums[i+1]){
                i++;
            }
        }
        return answer;
    }
    public List<List<Integer>> twoSum(int[] nums,int target,int start) {
        List<List<Integer>> answer = new LinkedList<>();
        int low = start;
        int high = nums.length-1;
        while (low<high){
            int sum = nums[low]+nums[high];
            int left = nums[low];
            int right = nums[high];
            if (sum<target){
                while (low<high&&nums[low]==left) {
                    low++;
                }
            }else if (sum>target){
                while (low<high&&nums[high]==right) {
                    high--;
                }
            }else if (sum==target){
                List<Integer> a = new LinkedList<Integer>();
                a.add(nums[low]);
                a.add(nums[high]);
                answer.add(a);
                while (low<high&&nums[low]==left) {
                    low++;
                }
                while (low<high&&nums[high]==right) {
                    high--;
                }
            }
        }
        return answer;
    }
    public int lengthOfLongestSubstring(String s) {
        if (s.length()==0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for(int i = 0; i < s.length(); i ++){
            if(map.containsKey(s.charAt(i))){
                left = Math.max(left,map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i),i);
            max = Math.max(max,i-left+1);
        }
        return max;

    }
    public boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        String input = s;
        int x = 0;
        int y = 0;
        for (int i = 0; i < input.length(); i++){

            if (input.charAt(x)==')'){
                if (!(stack.isEmpty())&&(stack.peekLast()=='(')){
                    stack.pollLast();
                    if (x!=input.length()-1){
                        x++;
                    }else {
                        y++;
                    }
                }
            }
            if (input.charAt(x)==']'){
                if (!(stack.isEmpty())&&stack.peekLast()=='['){
                    stack.pollLast();
                    if (x!=input.length()-1){
                        x++;
                    }else {
                        y++;
                    }
                }
            }
            if (input.charAt(x)=='}'){
                if (!(stack.isEmpty())&&stack.peekLast()=='{'){
                    stack.pollLast();
                    if (x!=input.length()-1){
                        x++;
                    }else {
                        y++;
                    }
                }
            }
            if (x==i&&y==0){
                stack.offerLast(input.charAt(x));
                x++;
            }
        }
        if (input.length() == 1) {
            return false;
        }else {
            return stack.isEmpty();
        }
    }
    public int closestCost(int[] baseCosts, int[] toppingCosts, int target) {
        boolean[] bag = new boolean[20001];
        int[] newtop = new int[toppingCosts.length*2];
        int a = 0;
        for (int i = 0; i < newtop.length; i++){
            newtop[i] = toppingCosts[a];
            i++;
            newtop[i] = toppingCosts[a];
            a++;
        }
        for (int base : baseCosts){
            bag[base] = true;
        }
        for (int top : newtop){
            for (int i = 20000; i >= top ; i--){
                bag[i] = bag[i-top]||bag[i];
            }
        }
        int minGAP = Integer.MAX_VALUE;
        int answer = 0;
        for (int i = 1; i < 20001; i++){
            if (bag[i]&&Math.abs(i-target)<minGAP){
                answer = i;
                minGAP = Math.abs(i-target);
            }
        }
        return answer;
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        Arrays.sort(candidates);
        Deque<Integer> path = new ArrayDeque<>();
        dfs2(candidates, 0, len, target, path, res);
        return res;
    }

    private void dfs2(int[] candidates, int begin, int len, int target, Deque<Integer> path, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = begin; i < len; i++) {
            if (target - candidates[i] < 0) {
                break;
            }

            path.addLast(candidates[i]);
            System.out.println("递归之前 => " + path + "，剩余 = " + (target - candidates[i]));

            dfs2(candidates, i, len, target - candidates[i], path, res);
            path.removeLast();
            System.out.println("递归之后 => " + path);
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);

        dfs1(nums, len, 0, path, used, res);
        return res;
    }

    private void dfs1(int[] nums, int len, int depth, Deque<Integer> path, boolean[] used, List<List<Integer>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;

                System.out.println("  递归之前 => " + path);
                dfs1(nums, len, depth + 1, path, used, res);

                used[i] = false;
                path.removeLast();
                System.out.println("递归之后 => " + path);
            }
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        int len = nums.length;

        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        res.add(new ArrayList<>());
        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);

        dfs(nums, len, 0, path, res, used);

        return res;
    }
    private void dfs(int[] nums, int len, int begin, Deque<Integer> path, List<List<Integer>> res, boolean[] used){
        for (int i = begin; i < len; i++){
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;
                System.out.println("递归之前 => " + path +"  i="+begin);
                dfs(nums, len, i, path, res, used);
                res.add(new ArrayList<>(path));
                used[i] = false;
                path.pollLast();
                System.out.println("递归之后 => " + path +"  i="+begin);
            }
        }
    }


    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        // 关键步骤
        Arrays.sort(candidates);

        Deque<Integer> path = new ArrayDeque<>(len);
        dfs(candidates, len, 0, target, path, res);
        return res;
    }

    /**
     * @param candidates 候选数组
     * @param len        冗余变量
     * @param begin      从候选数组的 begin 位置开始搜索
     * @param target     表示剩余，这个值一开始等于 target，基于题目中说明的"所有数字（包括目标数）都是正整数"这个条件
     * @param path       从根结点到叶子结点的路径
     * @param res
     */
    private void dfs(int[] candidates, int len, int begin, int target, Deque<Integer> path, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = begin; i < len; i++) {
            // 大剪枝：减去 candidates[i] 小于 0，减去后面的 candidates[i + 1]、candidates[i + 2] 肯定也小于 0，因此用 break
            if (target - candidates[i] < 0) {
                break;
            }

            // 小剪枝：同一层相同数值的结点，从第 2 个开始，候选数更少，结果一定发生重复，因此跳过，用 continue
            if (i > begin && candidates[i] == candidates[i - 1]) {
                continue;
            }

            path.addLast(candidates[i]);
            // 调试语句 ①
            System.out.println("递归之前 => " + path +"  i="+begin+"，剩余 = " + (target - candidates[i]));

            // 因为元素不可以重复使用，这里递归传递下去的是 i + 1 而不是 i
            dfs(candidates, len, i + 1, target - candidates[i], path, res);

            path.removeLast();
            // 调试语句 ②
            System.out.println("递归之后 => " + path +"  i="+begin+ "，剩余 = " + (target - candidates[i]));
        }
    }
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        int len = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        res.add(new ArrayList<>());
        // 关键步骤
        Arrays.sort(nums);
        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);
        dfs(nums, len, 0, used, path, res);
        return res;
    }
    private void dfs(int[] nums, int len, int begin, boolean[] used, Deque<Integer> path, List<List<Integer>> res){
        for (int i = begin; i < len; i++){
            if (i>0&&nums[i]==nums[i-1]&&!used[i-1]){
                continue;
            }
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;
                System.out.println("递归之前 => " + path +"  i="+begin);
                dfs(nums, len, i, used, path, res);
                res.add(new ArrayList<>(path));
                used[i] = false;
                path.pollLast();
                System.out.println("递归之后 => " + path +"  i="+begin);
            }
        }
    }
    public List<List<Integer>> permuteUnique(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);
        Arrays.sort(nums);
        dfs(nums, len, 0, path, used, res);

        return res;
    }

    private void dfs(int[] nums, int len, int depth, Deque<Integer> path, boolean[] used, List<List<Integer>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < len; i++) {
            if (i>0&&nums[i] == nums[i - 1]&&!used[i-1] ) {
                continue;
            }
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;

                System.out.println("  递归之前 => " + path);
                dfs(nums, len, depth + 1, path, used, res);

                used[i] = false;
                path.removeLast();
                System.out.println("递归之后 => " + path);
            }
        }
    }
    public List<String> restoreIpAddresses(String s) {
        int len = s.length();
        List<String> res = new ArrayList<>();
        // 如果长度不够，不搜索
        if (len < 4 || len > 12) {
            return res;
        }

        Deque<String> path = new ArrayDeque<>(4);
        int splitTimes = 0;
        dfs(s, len, splitTimes, 0, path, res);
        return res;
    }
    private void dfs(String s, int len, int split, int begin, Deque<String> path, List<String> res) {
        if (begin == len) {
            if (split == 4) {
                res.add(String.join(".", path));
            }
            return;
        }

        // 看到剩下的不够了，就退出（剪枝），len - begin 表示剩余的还未分割的字符串的位数
        if (len - begin < (4 - split) || len - begin > 3 * (4 - split)) {
            return;
        }

        for (int i = 0; i < 3; i++) {
            if (begin + i >= len) {
                break;
            }

            int ipSegment = judgeIfIpSegment(s, begin, begin + i);
            if (ipSegment != -1) {
                // 在判断是 ip 段的情况下，才去做截取
                path.addLast(ipSegment + "");
                dfs(s, len, split + 1, begin + i + 1, path, res);
                path.removeLast();
            }
        }
    }
    private int judgeIfIpSegment(String s, int left, int right) {
        int len = right - left + 1;

        // 大于 1 位的时候，不能以 0 开头
        if (len > 1 && s.charAt(left) == '0') {
            return -1;
        }

        // 转成 int 类型
        int res = Integer.parseInt(s.substring(left,right+1));

        if (res > 255) {
            return -1;
        }
        return res;
    }
    int y;
    int x;
    int c;
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        this.x = image[0].length;
        this.y = image.length;
        this.c = image[sr][sc];
        boolean[][] visited = new boolean[x][y];
        DFS(image, visited, sr, sc, newColor);
        return image;
    }
    private void DFS(int[][] image, boolean[][] visited, int sr, int sc, int color) {
        if (inArea(sr, sc) && !visited[sr][sc] && c == image[sr][sc]) {
            image[sr][sc] = color;
            visited[sr][sc] = true;
            DFS(image, visited, sr - 1, sc, color);
            DFS(image, visited, sr, sc - 1, color);
            DFS(image, visited, sr + 1, sc, color);
            DFS(image, visited, sr, sc + 1, color);
        }
    }
    private boolean inArea(int x, int y){
        return (x<this.x&&x>=0)&&(y>=0&&y<this.y);
    }
    boolean a = false;
    public boolean exist(char[][] board, String word) {
        this.x = board[0].length;
        this.y = board.length;
        boolean[][] visited = new boolean[y][x];
        Deque<String> path = new ArrayDeque<>();
        for (int i = 0; i < y; i++){
            for (int j = 0; j < x; j++){
                if (board[i][j]==word.charAt(0)&&!visited[i][j]){
                    DFS(board,visited,word, path,j,i);
                }
            }
        }
        return a;
    }
    private void DFS(char[][] board, boolean[][] visited, String word, Deque<String> path, int x, int y){
        if (word.equals(String.join("",path))){
            a = true;
            return;
        }
        if (a){
            return;
        }
        if (inArea(x,y)&&!visited[y][x]&&word.charAt(path.size())==board[y][x]){
            path.addLast(String.valueOf(word.charAt(path.size())));
            visited[y][x] = true;
            DFS(board,visited,word,path,x-1,y);
            DFS(board,visited,word,path,x,y-1);
            DFS(board,visited,word,path,x+1,y);
            DFS(board,visited,word,path,x,y+1);
            visited[y][x] = false;
            path.pollLast();
        }
    }
    public List<String> letterCombinations(String digits) {
        int len = digits.length();
        List<List<Character>> t = new ArrayList<>();
        for (int i = 0; i < len; i++){
            List<Character> p = new ArrayList<>();
            if (digits.charAt(i)=='2'){
                p.add('a');
                p.add('b');
                p.add('c');
            }else if (digits.charAt(i)=='3'){
                p.add('d');
                p.add('e');
                p.add('f');
            }else if (digits.charAt(i)=='4'){
                p.add('g');
                p.add('h');
                p.add('i');
            }else if (digits.charAt(i)=='5'){
                p.add('j');
                p.add('k');
                p.add('l');
            }else if (digits.charAt(i)=='6'){
                p.add('m');
                p.add('n');
                p.add('o');
            }else if (digits.charAt(i)=='7'){
                p.add('p');
                p.add('q');
                p.add('r');
                p.add('s');
            }else if (digits.charAt(i)=='8'){
                p.add('t');
                p.add('u');
                p.add('v');
            }else if (digits.charAt(i)=='9'){
                p.add('w');
                p.add('x');
                p.add('y');
                p.add('z');
            }
            t.add(p);
        }
        List<String> answer = new ArrayList<>();
        if (len==0){
            return answer;
        }
        int l = t.size();
        Deque<String> path = new ArrayDeque<>();
        DFSDIGITAL(path,answer,t,0,len,l);
        return answer;
    }
    private void DFSDIGITAL(Deque<String> path, List<String> answer, List<List<Character>> letter, int begin, int lenD,int lenL){
        if (path.size()==lenD){
            answer.add(String.join("",path));
            return;
        }
        for (int i = begin; i < lenL; i++){
            for (int j = 0; j < letter.get(i).size(); j++) {
                path.addLast(String.valueOf(letter.get(i).get(j)));
                DFSDIGITAL(path, answer, letter, i + 1, lenD, lenL);
                path.pollLast();
            }
        }
    }
    public List<String> generateParenthesis(int n) {
        Deque<String> path = new ArrayDeque<>();
        List<String> answer = new ArrayList<>();
        DFSParenthesis(path,answer,n*2);
        return answer;
    }
    int left = 0;
    int right = 0;
    private void DFSParenthesis(Deque<String> path, List<String> answer, int len){
        if (path.size()==len){
            answer.add(String.join("",path));
            return;
        }
        if (right>left){
            return;
        }
        if (left<len/2) {
            path.offerLast("(");
            left++;
            System.out.println("之前  "+path);
            DFSParenthesis(path,answer,len);
            System.out.println("之后  "+path);
            left--;
            path.pollLast();
        }
        if (right<len/2){
            path.offerLast(")");
            right++;
            System.out.println("之前  "+path);
            DFSParenthesis(path,answer,len);
            System.out.println("之后  "+path);
            right--;
            path.pollLast();
        }
    }
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> answer = new ArrayList<>();
        if (root == null){
            return answer;
        }
        Deque<TreeNode> level = new LinkedList<>();
        level.offerLast(root);
        int lev;
        boolean leftToRight = true;
        while (!level.isEmpty()){
            lev = level.size();
            List<Integer> layer = new ArrayList<>();
            for (int i = 0; i< lev; i++){
                TreeNode node = level.pollFirst();
                if (leftToRight){
                    layer.add(node.val);
                }else {
                    layer.add(0,node.val);
                }
                if (node.left!=null) {
                    level.offerLast(node.left);
                }
                if (node.right!=null) {
                    level.offerLast(node.right);
                }
            }
            answer.add(layer);
            leftToRight = !leftToRight;
        }
        return answer;
    }
    public int[][] generateMatrix(int n) {
        int l = 0, r = n - 1, t = 0, b = n - 1;
        int[][] mat = new int[n][n];
        int num = 1, tar = n * n;
        while(num <= tar){
            for(int i = l; i <= r; i++) {
                mat[t][i] = num++; // left to right.
            }
            t++;
            for(int i = t; i <= b; i++) {
                mat[i][r] = num++; // top to bottom.
            }
            r--;
            for(int i = r; i >= l; i--) {
                mat[b][i] = num++; // right to left.
            }
            b--;
            for(int i = b; i >= t; i--) {
                mat[i][l] = num++; // bottom to top.
            }
            l++;
        }
        return mat;
    }
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode pre = dummy;
        ListNode end = dummy;

        while (end.next != null) {
            for (int i = 0; i < k && end != null; i++) {
                end = end.next;
            }
            if (end == null) {
                ListNode start = pre.next;
                ListNode next = null;
                pre.next = reverse(start);
                start.next = null;
                break;
            }
            ListNode start = pre.next;
            ListNode next = end.next;
            end.next = null;
            pre.next = reverse(start);
            start.next = next;
            pre = start;

            end = pre;
        }
        return dummy.next;
    }

    private ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }
    public boolean canJump(int[] nums) {
        int length = nums.length;
        if (length==1){
            return true;
        }
        boolean[] dp = new boolean[length+1];
        dp[1] = true;
        for (int i = 1; i < length; i++){
            if (nums[i-1]==0){
                dp[i] = false;
                continue;
            }
            if (dp[i]) {
                for (int j = 0; j <= nums[i - 1]; j++) {
                    if (i + j < length + 1) {
                        dp[i + j] = true;
                    }
                }
            }
        }
        return dp[length];
    }
    public int[] maxSlidingWindow(int[] nums, int k) {
        Queue<Integer> q = new PriorityQueue<>(cmp);
        int left = 0;
        int right = k-1;
        int length = nums.length;
        ArrayList<Integer> answer = new ArrayList<>();
        if (length==1){
            return nums;
        }
        for (int i = 0; i < k-1; i++){
            q.offer(nums[i]);
        }
        for (;right<length; right++){
            q.offer(nums[right]);
            answer.add(q.peek());
            q.remove(nums[left]);
            left++;
        }
        int[] a = answer.stream().mapToInt(Integer::valueOf).toArray();
        return a;
    }
    static Comparator<Integer> cmp = new Comparator<Integer>() {
        public int compare(Integer e1, Integer e2) {
            return e2 - e1;
        }
    };
    public int minElements(int[] nums, int limit, int goal) {
        long sum = 0;
        for (long n : nums){
            sum+=n;
        }

        sum = goal - sum;
        if (sum==0){
            return 0;
        }
        if (Math.abs(sum)<=limit){
            return 1;
        }else {
            long x = (Math.abs(sum)/limit);
            if (Math.abs(sum)%limit==0){
                return (int) x;
            }
            return (int) (x+1);
        }
    }
    public int NumOfWays(int n) {
        if (n == 0) {
            return 0;
        }else if (n == 1) {
            return 12;
        }
        long temp = 1000000007;
        long  repeat = 6;
        long  unrepeat = 6;
        for(int i = 2; i <=n; i++) {
            long  newrep = (repeat * 3) % temp + unrepeat * 2 % temp;
            long  newunrep = repeat * 2 % temp + unrepeat * 2 % temp;
            repeat = newrep;
            unrepeat = newunrep;
        }
        return (int)((repeat + unrepeat)%temp);
    }
    public int leastInterval(char[] tasks, int n) {
        int[] buckets = new int[26];
        for (char task : tasks) {
            buckets[task - 'A']++;
        }
        Arrays.sort(buckets);
        int maxTimes = buckets[25];
        int maxCount = 1;
        for(int i = 25; i >= 1; i--){
            if(buckets[i] == buckets[i - 1]) {
                maxCount++;
            }else {
                break;
            }
        }
        int res = (maxTimes - 1) * (n + 1) + maxCount;
        return Math.max(res, tasks.length);
    }
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int num : nums) {
            int x = (num - 1)%n;
            nums[x] += n;
        }
        List<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                ret.add(i + 1);
            }
        }
        return ret;
    }
    public int subarraySum(int[] nums, int k) {
        // key：前缀和，value：key 对应的前缀和的个数
        Map<Integer, Integer> preSumFreq = new HashMap<>();
        // 对于下标为 0 的元素，前缀和为 0，个数为 1
        preSumFreq.put(0, 1);

        int preSum = 0;
        int count = 0;
        for (int num : nums) {
            preSum += num;

            // 先获得前缀和为 preSum - k 的个数，加到计数变量里
            if (preSumFreq.containsKey(preSum - k)) {
                count += preSumFreq.get(preSum - k);
            }

            // 然后维护 preSumFreq 的定义
            preSumFreq.put(preSum, preSumFreq.getOrDefault(preSum, 0) + 1);
        }
        return count;
    }
    public int majorityElement(int[] nums) {
        int count = 0;
        int candidate = 0;
        for (int n : nums){
            if (count==0){
                candidate = n;
            }
            if (n==candidate) {
                count++;
            }else {
                count--;
            }
        }
        return candidate;
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length-1;
        int col = 0;

        while (row >= 0 && col < matrix[0].length) {
            if (matrix[row][col] > target) {
                row--;
            } else if (matrix[row][col] < target) {
                col++;
            } else { // found it
                return true;
            }
        }
        return false;
    }

    public double maxAverageRatio(int[][] classes, int extraStudents) {
        int len = classes.length;
        PriorityQueue<BigDecimal> s = new PriorityQueue<>();
        BigDecimal sum = new BigDecimal(0);
        BigDecimal[] o = new BigDecimal[len];
        for (int i = 0; i < len; i++){
            BigDecimal temp = BigDecimal.valueOf(classes[i][0] * 1.0 / classes[i][1]);
            o[i] = temp;
            s.add(temp);
            sum = sum.add(temp);
        }
        System.out.println(sum.divide(BigDecimal.valueOf(len)));
        for (int i = 0; i < extraStudents; i++){
            BigDecimal t = s.poll();
            for (int a = 0; a < len; a++){
                if (o[a].equals(t)){
                    BigDecimal t1 = BigDecimal.valueOf((classes[a][0] + 1) * 1.0 / (classes[a][1] + 1));
                    classes[a][0] += 1;
                    classes[a][1] += 1;
                    sum = sum.add(t1.subtract(t));
                    o[a] = t1;
                    s.add(t1);
                    break;
                }
            }
        }
        String res = sum.toString();
        double ans = Double.parseDouble(res);
        return ans/len;
    }
    public int maximumScore(int[] nums, int k) {
        int i = k;
        int j = k;
        int len = nums.length;
        int res = 0;
        while (true){
            while (i>=0&&nums[i]>=nums[k]){
                i--;
            }
            while (j<len&&nums[j]>=nums[k]){
                j++;
            }
            res = Math.max(res,(j-i-1)*nums[k]);
            if (i>=0&&j<len){
                nums[k] = Math.max(nums[i],nums[j]);
            }else if (i<0&&j>=len){
                break;
            }else if (i<0){
                nums[k] = nums[j];
            }else {
                nums[k] = nums[i];
            }
        }
        return res;
    }
    public int maxResult(int[] nums, int k) {
        int len = nums.length;
        int[] dp = new int[len + 1];
        Arrays.fill(dp, Integer.MIN_VALUE);
        dp[1] = nums[0];
        for (int i = 1; i < len + 1; i++) {
            if (i + k < len) {
                for (int j = i + 1; j <= i + k; j++) {
                    dp[j] = Math.max(dp[j], nums[j - 1] + dp[i]);
                }
            } else {
                for (int j = i + 1; j < len + 1; j++) {
                    dp[j] = Math.max(dp[j], nums[j - 1] + dp[i]);
                }
            }

        }
        return dp[len];
    }
    final int MAX = Integer.MAX_VALUE;
    public int minJumps(int[] arr) {
        int n = arr.length;
        Queue<Integer> queue = new LinkedList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int a = arr[i];
            List<Integer> list = map.getOrDefault(a, new ArrayList<>());
            list.add(i);
            map.put(a, list);
        }
        int[] dis = new int[n]; //每个元素到最后一个元素的距离，用来存结果
        Arrays.fill(dis, MAX);
        dis[n - 1] = 0;
        queue.offer(n - 1);
        boolean[] visited = new boolean[n]; //记录元素是否被访问过，初始都是false
        while (!queue.isEmpty()) {
            int x = queue.poll();
            if (x - 1 >= 0 && dis[x - 1] == MAX) { //等于MAX说明x左边的元素还没有计算过
                dis[x - 1] = dis[x] + 1;
                queue.offer(x - 1);
            }
            if (x + 1 < n && dis[x + 1] == MAX) {
                dis[x + 1] = dis[x] + 1;
                queue.offer(x + 1);
            }
            if (!visited[x]) {
                for (int i : map.get(arr[x])) {
                    if (dis[i] == MAX) {
                        dis[i] = dis[x] + 1;
                        queue.offer(i);
                        visited[i] = true;
                    }
                }
            }
        }
        return dis[0];
    }
    public int maxSumDivThree(int[] nums) {
        int len = nums.length;
        int[][] dp = new int[len][3];
        int firstNum = nums[0];
        if(firstNum%3==0){
            dp[0][0] = firstNum;
            dp[0][1] = Integer.MIN_VALUE;
            dp[0][2] = Integer.MIN_VALUE;
        }else {
            dp[0][0] = 0;
            dp[0][1] = Integer.MIN_VALUE;
            dp[0][2] = Integer.MIN_VALUE;
            dp[0][firstNum%3] = firstNum;
        }

        for (int i = 1; i < len; i++) {
            int curNum = nums[i];
            if (curNum % 3 == 0) {
                dp[i][0] = Math.max(dp[i - 1][0] + curNum, dp[i - 1][0]);
                dp[i][1] = Math.max(dp[i - 1][1] + curNum, dp[i - 1][1]);
                dp[i][2] = Math.max(dp[i - 1][2] + curNum, dp[i - 1][2]);
            } else if (curNum % 3 == 1) {
                dp[i][0] = Math.max(dp[i - 1][2] + curNum, dp[i - 1][0]);
                dp[i][1] = Math.max(dp[i - 1][0] + curNum, dp[i - 1][1]);
                dp[i][2] = Math.max(dp[i - 1][1] + curNum, dp[i - 1][2]);
            } else if (curNum % 3 == 2) {
                dp[i][0] = Math.max(dp[i - 1][1] + curNum, dp[i - 1][0]);
                dp[i][1] = Math.max(dp[i - 1][2] + curNum, dp[i - 1][1]);
                dp[i][2] = Math.max(dp[i - 1][0] + curNum, dp[i - 1][2]);
            }
        }
        return dp[len-1][0];
    }
    public int removeElement(int[] nums, int val) {
        int left = 0;
        for (int right = 0; right < nums.length; right++){
            if (nums[right]==val){
                continue;
            }else {
                nums[left] = nums[right];
                left++;
            }
        }
        return left;
    }
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return 1;
        }
        int left = 0;
        for (int right = 0; right < nums.length; right++) {
            if (nums[right] != nums[left]) {
                left++;
                nums[left] = nums[right];

            }
        }
        return left;
    }
    public int strStr(String haystack, String needle) {
        if (needle.length()==0){
            return 0;
        }
        char[] haystackC = haystack.toCharArray();
        char[] needleC = needle.toCharArray();
        int[] prefixTable = new int[needle.length()];
        prefixTable(needleC,prefixTable,needle.length());
        int i = 0;
        int j = 0;
        while (i < haystack.length()){

            if (haystackC[i]==needleC[j]){
                i++;
                j++;
                if (j==needle.length()){
                    return i-j;
                }
            }else {
                j = prefixTable[j];
                if (j==-1){
                    i++;
                    j++;
                }
            }
        }
        return 0;
    }
    public void prefixTable(char[] needle, int[] prefixTable, int needleLength){
        prefixTable[0] = 0;
        int length = 0;
        int i = 1;
        while (i < needleLength){
            if (needle[i] == needle[length]){
                length++;
                prefixTable[i] = length;
                i++;
            }else {
                if (length > 0){
                    length = prefixTable[length-1];
                }else {
                    prefixTable[i] = length;
                    i++;
                }
            }
        }
        if (needleLength - 1 >= 0) {
            System.arraycopy(prefixTable, 0, prefixTable, 1, needleLength - 1);
        }
        prefixTable[0] = -1;
    }
    public int numDecodings(String s) {
        String[] coin = new String[100];
        Arrays.fill(coin,"9999");
        for (int i = 1; i < 27; i++){
            coin[i] = String.valueOf(i);
        }
        char[] tgt = s.toCharArray();
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = 0;
        for (int i = 1; i < s.length()+1; i++){
                String one = String.valueOf(tgt[i-1]);
                String two = "";
                if (i>1) {
                    two = String.valueOf(tgt[i - 2]) + String.valueOf(tgt[i - 1]);
                }
                if (i>1&&coin[Integer.parseInt(one)].equals(one)&&coin[Integer.parseInt(two)].equals(two)){
                    dp[i] = dp[i-1]+dp[i-2];
                }else if (coin[Integer.parseInt(one)].equals(one)){
                    dp[i] = dp[i-1];
                }else if (i>1&&coin[Integer.parseInt(two)].equals(two)){
                    dp[i] = dp[i-2];
                }
        }
        return dp[s.length()];
    }
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int[][] sum = new int[matrix.length + 1][matrix[0].length + 1];
        int res = Integer.MIN_VALUE;
        for (int i = 1; i <= matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                sum[i][j] = sum[i][j - 1] + sum[i - 1][j] - sum[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
        int cur = Integer.MIN_VALUE;
        for (int x = 1; x <= matrix[0].length; x++){
            for (int y = 1; y <= matrix.length; y++){
                for (int i = 1; i <= x; i++){
                    for (int j = 1; j <= y; j++){
                        cur = sum[y][x] - sum[j - 1][x] - sum[y][i - 1] + sum[j - 1][i - 1];
                        if (cur==k){
                            return k;
                        }
                        if (cur>k){
                            continue;
                        }
                        res = Math.max(res,cur);
                    }
                }
            }
        }
        return res;
    }
    public List<Integer> largestDivisibleSubset(int[] nums) {
        int length = nums.length;
        Arrays.sort(nums);
        int[] dp = new int[length];
        int[] backtrack = new int[length];
        List<Integer> answer = new ArrayList<>();
        for (int i = 0; i < length; i++){
            int len = 1;
            int prevIndex = i;
            for (int j = i-1; j >=0; j--){
                if (nums[i]%nums[j]==0){
                    if (dp[j]+1>len){
                        len = dp[j]+1;
                        prevIndex = j;
                    }
                }
            }
            dp[i] = len;
            backtrack[i] = prevIndex;
        }
        int max = dp[0];
        int index = 0;
        for (int i = 0; i < length; i++){
            if (dp[i]>max){
                max = dp[i];
                index = i;
            }
        }
        while (answer.size()<max){
            answer.add(nums[index]);
            index = backtrack[index];
        }
        return answer;
    }

    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        dp[0] = 1;
        for (int i = 1; i < target+1; i++){
            for (int j = 0; j < nums.length; j++){
                if (i-nums[j]>=0) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        return dp[target];
    }
    private TreeNode resNode;

    public TreeNode increasingBST(TreeNode root) {
        TreeNode dummyNode = new TreeNode(-1);
        resNode = dummyNode;
        inorder(root);
        return dummyNode.right;
    }

    public void inorder(TreeNode node) {
        if (node == null) {
            return;
        }
        inorder(node.left);

        // 在中序遍历的过程中修改节点指向
        resNode.right = node;
        node.left = null;
        resNode = node;

        inorder(node.right);
    }
    public int shipWithinDays(int[] weights, int D) {
        int max = weights[0];
        int sum = 0;
        for (int n : weights){
            sum += n;
            if (n>max){
                max = n;
            }
        }

        while (max<sum){
            int middle = max + (sum - max) / 2;
            if (canShip(weights,D,middle)){
                sum = middle;
            }else{
                max = middle+1;
            }
        }
        return max;
    }
    private boolean canShip(int[] weights, int D, int ship){
        int temp = 1;
        int sum = 0;
        for (int i = 0; i < weights.length; i++){
            sum += weights[i];
            if (ship < sum){
                sum = weights[i];
                temp++;
            }
        }
        return temp<=D;
    }
    int sumBST = 0;
    public int rangeSumBST(TreeNode root, int low, int high) {
        inorderSum(root,low,high);
        return sumBST;
    }
    public void inorderSum(TreeNode node, int low, int high) {
        if (node == null) {
            return;
        }
        inorderSum(node.left,low,high);

        if (node.val>=low&&node.val<=high){
            sumBST += node.val;
        }

        inorderSum(node.right,low,high);
    }
    HashMap<Integer,Integer> stone = new HashMap<>();
    HashMap<String,Boolean> cache = new HashMap<>();
    public boolean canCross(int[] stones) {
        int length = stones.length;
        for (int i = 0; i < length; i++){
            stone.put(stones[i],i);
        }
        if (!stone.containsKey(1)){
            return false;
        }
        return toadDFS(stones,length,1,1);
    }
    private boolean toadDFS(int[] stones, int length, int index, int k){
        if (index == length - 1){
            return true;
        }
        String key = index + "-" + k;
        if (cache.containsKey(key)){
            return cache.get(key);
        }
        for (int i = -1; i < 2; i++){
            if (k+i<=0){
                continue;
            }
            int temp = k + i + stones[index];
            if (stone.containsKey(temp)){
                boolean tempb = toadDFS(stones,length,stone.get(temp),k+i);
                if (tempb){
                    cache.put(key,true);
                    return true;
                }
            }
        }
        cache.put(key,false);
        return false;
    }
    public boolean canCrossDP(int[] stones) {
        int n = stones.length;
        // check first step
        if (stones[1] != 1) return false;
        boolean[][] f = new boolean[n][n];
        f[1][1] = true;
        for (int i = 2; i < n; i++) {
            for (int j = 1; j < i; j++) {
                int k = stones[i] - stones[j];
                if (k<=j+1) {
                    f[i][k] = f[j][k - 1] || f[j][k] || f[j][k + 1];
                }
            }
        }
        for (int i = 1; i < n; i++) {
            if (f[n - 1][i]) return true;
        }
        return false;
    }
    public int singleNumber(int[] nums) {
        int[] cnt = new int[32];
        for (int x : nums) {
            for (int i = 0; i < 32; i++) {
                if (((x >> i) & 1) == 1) {
                    cnt[i]++;
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            if ((cnt[i] % 3 & 1) == 1) {
                ans += (1 << i);
            }
        }
        return ans;
    }

    public int[] decode(int[] encoded, int first) {
        int[] decode = new int[encoded.length+1];
        decode[0] = first;
        for (int i = 1;   i < decode.length; i++){
            decode[i] = encoded[i-1]^decode[i-1];
        }
        return decode;
    }

    public int xorOperation(int n, int start) {
        // 结果的最低位
        // 当n和start的最低位都为1时，lowestOrder=1，否则为0
        int lowestOrder = n & start & 1;

        // start ^ (start+2) ^ (start+4) ^ …… ^(start + 2*(n-1))
        //  =(令s=start/2)
        // (s ^ (s+1) ^ (s+2) ^ …… ^ (s+n-1)) * 2 + lowestOrder
        // 此处lowestOrder是为了补全start/2时丢失的1
        int s = start / 2;

        // 而n到m(n<m)的异或等于1到n-1的异或 异或 1到m的异或
        // 原因：a^a = 0   0^a = a
        int result = computeXOR(s - 1) ^ computeXOR(s + n - 1);

        return result * 2 + lowestOrder;
    }
    int computeXOR(int n) {
        // 前n个数异或的结果是有规律的
        // 例如：   二进制数   异或结果    return
        //      1    0001    0001        1
        //      2    0010    0011       n+1
        //      3    0011    0000        0
        //      4    0100    0100        n
        //      5    0101    0001        1
        //      6    0110    0111       n+1
        //      ……   ……      ……
        switch(n % 4)
        {
            case 0:
                return n;
            case 1:
                return 1;
            case 2:
                return n + 1;
            // case3
            default:
                return 0;
        }
    }
    int[] jobs;
    int n, k;
    int ansj = 0x3f3f3f3f;
    public int minimumTimeRequired(int[] _jobs, int _k) {
        jobs = _jobs;
        n = jobs.length;
        k = _k;
        int[] sum = new int[k];
        dfs(0, 0, sum, 0);
        return ansj;
    }
    /**
     * u     : 当前处理到那个 job
     * used  : 当前分配给了多少个工人了
     * sum   : 工人的分配情况          例如：sum[0] = x 代表 0 号工人工作量为 x
     * max   : 当前的「最大工作时间」
     */
    void dfs(int u, int used, int[] sum, int max) {
        if (max >= ansj) return;
        if (u == n) {
            ansj = max;
            return;
        }
        // 优先分配给「空闲工人」
        if (used < k) {
            sum[used] = jobs[u];
            dfs(u + 1, used + 1, sum, Math.max(sum[used], max));
            sum[used] = 0;
        }
        for (int i = 0; i < used; i++) {
            sum[i] += jobs[u];
            dfs(u + 1, used, sum, Math.max(sum[i], max));
            sum[i] -= jobs[u];
        }
    }
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int len = nums.length;
        if (len == 0) {
            return res;
        }

        // 使用二分搜索树方便排序
        Set<Integer> set = new TreeSet();
        for (int i = 0; i < len; i++) {
            set.add(nums[i]);
        }

        // 排名表
        Map<Integer, Integer> map = new HashMap<>();
        int rank = 1;
        for (Integer num : set) {
            map.put(num, rank);
            rank++;
        }

        FenwickTree fenwickTree = new FenwickTree(set.size() + 1);
        // 从后向前填表
        for (int i = len - 1; i >= 0; i--) {
            // 1、查询排名
            rank = map.get(nums[i]);
            // 2、在树状数组排名的那个位置 + 1
            fenwickTree.update(rank, 1);
            // 3、查询一下小于等于“当前排名 - 1”的元素有多少
            res.add(fenwickTree.query(rank - 1));
        }
        Collections.reverse(res);
        return res;
    }

    private static class FenwickTree {
        private int[] tree;
        private int len;

        public FenwickTree(int n) {
            this.len = n;
            tree = new int[n + 1];
        }
        // 单点更新：将 index 这个位置 + 1
        public void update(int i, int delta) {
            // 从下到上，最多到 size，可以等于 size
            while (i <= this.len) {
                tree[i] += delta;
                i += lowbit(i);
            }
        }
        // 区间查询：查询小于等于 index 的元素个数
        // 查询的语义是"前缀和"
        public int query(int i) {
            // 从右到左查询
            int sum = 0;
            while (i > 0) {
                sum += tree[i];
                i -= lowbit(i);
            }
            return sum;
        }

        public int lowbit(int x) {
            return x & (-x);
        }
    }
    public int minDays(int[] bloomDay, int m, int k) {
        if (m*k>bloomDay.length){
            return -1;
        }
        int minDay = 0x3f3f3f3f;
        int maxDay = -1;
        for (int num : bloomDay){
            if (maxDay<num){
                maxDay = num;
            }
            if (minDay>num){
                minDay = num;
            }
        }
        while (minDay<maxDay){
            int middle = minDay + (maxDay - minDay)/2;
            if (canBloom(middle,m,k,bloomDay)){
                maxDay = middle;
            }else {
                minDay = middle+1;
            }
        }
        return minDay;
    }
    private boolean canBloom(int middle, int m, int k, int[] flowers){
        int count = 0;
        for (int i = 0; i < flowers.length; i++){
            if (flowers[i]<=middle){
                count++;
                int j = i;
                if (j+k<=flowers.length){
                    for (; j < i+k; j++){
                        if (flowers[j]>middle){
                            count--;
                            break;
                        }
                    }
                    i = j-1;
                }else {
                    count--;
                    break;
                }
            }
        }
        return count>=m;
    }
    public int maximumPopulation(int[][] logs) {
        int[] years = new int[2051];
        for (int[] log : logs) {
            for (int j = log[0]; j < log[1]; j++) {
                years[j]++;
            }
        }
        int ans = 0;
        int index = 0;
        for (int i = 1949; i < 2051; i++){
            if (years[i]>ans){
                ans = years[i];
                index = i;
            }
        }
        return index;
    }
    public int maxDistance(int[] nums1, int[] nums2) {
        int n1len = nums1.length;
        int n2len = nums2.length;

        if (nums1[n1len-1]>=nums2[0]){
            if (nums1[n1len-1]==nums2[0]) {
                int ans = 0;
                for (int i = 1; i < n2len; i++) {
                    if (nums2[i] != nums2[i - 1]) {
                        return ans;
                    } else {
                        ans++;
                    }
                }
            }
            return 0;
        }
        if (nums2[n2len-1]>=nums1[0]){
            return n2len-1;
        }
        int result = 0;
        for (int i = n2len-1; i >=0;i--){
            int low = 0;
            int high = n1len-1;
            int middle;
            while (low<high){
                middle = (low + high)/2;
                if (nums1[middle]<=nums2[i]){
                    high = middle;
                }
                if (nums1[middle]>nums2[i]){
                    low = middle+1;
                }
            }
            result = Math.max(result,i-low);
        }
        return result;
    }
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        if (len == 0) {
            return 0;
        }

        if (len == 1) {
            return heights[0];
        }

        int res = 0;

        int[] newHeights = new int[len + 2];
        newHeights[0] = 0;
        System.arraycopy(heights, 0, newHeights, 1, len);
        newHeights[len + 1] = 0;
        len += 2;
        heights = newHeights;

        Deque<Integer> stack = new ArrayDeque<>(len);
        // 先放入哨兵，在循环里就不用做非空判断
        stack.addLast(0);

        for (int i = 1; i < len; i++) {
            while (heights[i] < heights[stack.peekLast()]) {
                int curHeight = heights[stack.pollLast()];
                int curWidth = i - stack.peekLast() - 1;
                res = Math.max(res, curHeight * curWidth);
            }
            stack.addLast(i);
        }
        return res;
    }

    public int maxSumMinProduct(int[] nums) {
        int len = nums.length;

        int[] lefti = new int[len];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < len; i++) {
            while (!stack.empty() && nums[stack.peek()] >= nums[i]) {
                stack.pop();
            }
            if (stack.empty()) {
                lefti[i] = -1;
            } else {
                lefti[i] = stack.peek();
            }
            stack.push(i);
        }

        int[] righti = new int[len];
        stack.clear();
        for (int i = len - 1; i >= 0; i--) {
            while (!stack.empty() && nums[stack.peek()] >= nums[i]) {
                stack.pop();
            }
            if (stack.empty()) {
                righti[i] = len;
            } else {
                righti[i] = stack.peek();
            }
            stack.push(i);
        }

        long[] preSum = new long[len + 1];
        for (int i = 1; i <= len; i++) {
            preSum[i] = nums[i - 1] + preSum[i - 1];
        }

        long res = 0;
        for (int i = 0; i < len; i++) {
            res = Math.max(res, nums[i] * (preSum[righti[i]] - preSum[lefti[i] + 1]));
        }
        return (int)(res % (1000000007));
    }
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        String tree1 = leaf(root1,"");
        String tree2 = leaf(root2,"");
        return tree1.equals(tree2);
    }
    private String leaf(TreeNode root,String leaves){
        if (root==null){
            return leaves;
        }
        if (root.left==null&&root.right==null){
            leaves += root.val;
        }
        return leaf(root.left,leaves)+leaf(root.right,leaves);
    }
    public int[] decode(int[] encoded) {
        int[] decode = new int[encoded.length+1];
        int x = computeXOR(encoded.length+1);
        int en = encoded[1];
        for (int i = 3; i < encoded.length; i=i+2){
            en = en^encoded[i];
        }
        decode[0] = x^en;
        for (int i = 1;   i < decode.length; i++){
            decode[i] = encoded[i-1]^decode[i-1];
        }
        return decode;
    }
    public int[] xorQueries(int[] arr, int[][] queries) {
        int[] result = new int[queries.length];
        int[] prefixXOR = new int[arr.length+1];
        prefixXOR[0] = 0;
        for (int i = 1; i < arr.length+1; i++){
            prefixXOR[i] = prefixXOR[i-1]^arr[i-1];
        }

        for (int i = 0; i < queries.length; i++){
            result[i] = prefixXOR[queries[i][0]]^prefixXOR[queries[i][1]+1];
        }
        return result;
    }
    public int numWays(int steps, int arrLen) {
        int mod = 1000000007;
        int max = Math.min(steps/2,arrLen-1);
        long[][] dp = new long[max+1][steps+1];
        dp[0][steps] = 1;
        for (int i = steps-1; i >= 0; i--){
            for (int j = 0; j<max+1;j++){
                if (j-1<0){
                    dp[j][i] = (dp[j+1][i+1] + dp[j][i+1])%mod;
                }else if (j+1>max){
                    dp[j][i] = (dp[j-1][i+1] + dp[j][i+1])%mod;
                }else {
                    dp[j][i] = (dp[j+1][i+1] + dp[j-1][i+1] + dp[j][i+1])%mod;
                }
            }
        }
        return (int) dp[0][0];
    }
    public String intToRoman(int num) {
        HashMap<Integer,String> intToRoman = new HashMap<>();
        intToRoman.put(1,"I");
        intToRoman.put(4,"IV");
        intToRoman.put(5,"V");
        intToRoman.put(9,"IX");
        intToRoman.put(10,"X");
        intToRoman.put(40,"XL");
        intToRoman.put(50,"L");
        intToRoman.put(90,"XC");
        intToRoman.put(100,"C");
        intToRoman.put(400,"CD");
        intToRoman.put(500,"D");
        intToRoman.put(900,"CM");
        intToRoman.put(1000,"M");
        String answer ="";
        for (int i = 0; i < 4; i++){
            int temp = num%10;
            num = num/10;
            if (temp==0&&num==0){
                break;
            }
            temp = (int) (temp*Math.pow(10,i));
            if (intToRoman.containsKey(temp)){
                answer = intToRoman.get(temp) + answer;
            }else {
                if (temp>5*(int)Math.pow(10,i)){
                    int check = 5*(int)Math.pow(10,i);
                    String tem = intToRoman.get(5*(int)Math.pow(10,i));
                    while (check!=temp){
                        check += 1*(int)Math.pow(10,i);
                        tem = tem + intToRoman.get(1*(int)Math.pow(10,i));
                    }
                    answer = tem + answer;
                }else {
                    int check = 0;
                    String tem = "";
                    while (check!=temp){
                        check += 1*(int)Math.pow(10,i);
                        tem = tem + intToRoman.get(1*(int)Math.pow(10,i));
                    }
                    answer = tem + answer;
                }
            }
        }
        return answer;
    }
    public int romanToInt(String s) {
        HashMap<String,Integer> romanToInt = new HashMap<>();
        romanToInt.put("I",1);
        romanToInt.put("V",5);
        romanToInt.put("X",10);
        romanToInt.put("L",50);
        romanToInt.put("C",100);
        romanToInt.put("D",500);
        romanToInt.put("M",1000);
        int answer = 0;
        for (int i = s.length()-1; i>=0 ; i--){
            if (s.charAt(i)=='I'&&answer>=4){
                answer = answer-1;
                continue;
            }
            if (s.charAt(i)=='X'&&answer>=40){
                answer = answer-10;
                continue;
            }
            if (s.charAt(i)=='C'&&answer>=400){
                answer = answer-100;
                continue;
            }
            answer = answer + romanToInt.get(String.valueOf(s.charAt(i)));
        }
        return answer;
    }

    public int subsetXORSum(int[] nums) {
        int answer = 0;
        int len = nums.length;

        for (int i = 0; i < (1 << len); i++) {
            int res = 0;
            for (int j = 0; j < len; j++) {
                if ((i & (1 << j))==(1 << j)) {
                    res = res ^ nums[j];
                }
            }
            answer += res;
        }
        return answer;
    }
    public int minSwaps(String s) {
        int sum0 = 0;
        int sum1 = 0;
        int odd0 = 0;
        int odd1 = 0;

        for (int i = 0; i < s.length(); i++){
            if (s.charAt(i)=='1'){
                sum1++;
                if (i%2==0){
                    odd1++;
                }
            }else {
                sum0++;
                if (i%2==0){
                    odd0++;
                }
            }
        }

        if (s.length()%2==0){
            if (sum1!=sum0){
                return -1;
            }
            return Math.min(odd0,odd1);
        }else {
            if (Math.abs(sum1-sum0)!=1){
                return -1;
            }
            if (sum1>sum0){
                return odd0;
            }else {
                return odd1;
            }
        }
    }
    HashMap<Integer,int[]> twoArray;
    HashMap<Integer,Integer> sums;
    public void FindSumPairs(int[] nums1, int[] nums2) {
        twoArray = new HashMap<>();
        sums = new HashMap<>();
        for (int i = 0; i < nums2.length; i++){
            int[] temp = new int[nums1.length];
            for (int j = 0; j < nums1.length; j++){
                temp[j] = nums2[i] + nums1[j];
                sums.put(nums2[i] + nums1[j],sums.getOrDefault(nums2[i] + nums1[j],0)+1);
            }
            twoArray.put(i,temp);
        }
    }

    public void add(int index, int val) {
        int[] temp = twoArray.get(index);
        for (int i = 0; i < temp.length; i++){
            sums.replace(temp[i],sums.get(temp[i])-1);
            temp[i] += val;
            if (sums.containsKey(temp[i])){
                sums.replace(temp[i],sums.get(temp[i])+1);
            }else {
                sums.put(temp[i],1);
            }
        }
        twoArray.replace(index,temp);
    }

    public int count(int tot) {
        if (sums.containsKey(tot)) {
            return sums.get(tot);
        }
        return 0;
    }

    public boolean isCousins(TreeNode root, int x, int y) {
        Queue<TreeNode> stackx = new LinkedList<>();
        stackx.offer(root);
        int nx = 0;
        int ny = 0;
        while (!stackx.isEmpty()) {
            int size = stackx.size();
            for (int i = 0; i<size; i++){
                TreeNode node = stackx.poll();
                if (node==null){
                    continue;
                }
                if (node.left!=null&&node.left.val==x){
                    nx++;
                }
                if (node.left!=null&&node.left.val==y){
                    ny++;
                }
                if (node.right!=null&&node.right.val==y){
                    ny++;
                }
                if (node.right!=null&&node.right.val==x){
                    nx++;
                }
                if (nx==1&&ny==1){
                    if (node.right != null&&node.left!=null) {
                        return !((node.right.val==y||node.right.val==x)&&(node.left.val==x||node.left.val==y));
                    }else {
                        return true;
                    }
                }
                stackx.offer(node.left);
                stackx.offer(node.right);
            }
            nx=0;
            ny=0;
        }
        return false;
    }
    public int countTriplets(int[] arr) {

        int[] prefixXOR = new int[arr.length+1];
        for (int i = 1; i < arr.length + 1; i++){
            prefixXOR[i] = prefixXOR[i-1]^arr[i-1];
        }
        int answer = 0;
        for (int j = 0; j < arr.length; j++){
            for (int i = 0; i < j; i++){
                if (prefixXOR[i]==prefixXOR[j+1]){
                    answer += j-i;
                }
            }
        }
        return answer;
         /*
        int n = arr.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            s[i + 1] = s[i] ^ arr[i];
        }
        Map<Integer, Integer> cnt = new HashMap<Integer, Integer>();
        Map<Integer, Integer> total = new HashMap<Integer, Integer>();
        int ans = 0;
        for (int k = 0; k < n; ++k) {
            cnt.put(s[k], cnt.getOrDefault(s[k], 0) + 1);
            total.put(s[k], total.getOrDefault(s[k], 0) + k);
            if (cnt.containsKey(s[k + 1])) {
                ans += cnt.get(s[k + 1]) * k - total.get(s[k + 1]);
            }

        }
        return ans;
        */
    }
    public int kthLargestValue(int[][] matrix, int k) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] prefixXOR = new int[m + 1][n + 1];
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for (int i = 1; i < m+1; i++) {
            for (int j = 1; j < n+1; j++) {
                prefixXOR[i][j] = prefixXOR[i][j - 1] ^ prefixXOR[i - 1][j] ^ prefixXOR[i - 1][j - 1] ^ matrix[i - 1][j - 1];
                if(queue.size()<k){
                    queue.add(prefixXOR[i][j]);
                }else{
                    int maxInQueue = queue.peek();
                    if(maxInQueue<prefixXOR[i][j]){
                        queue.poll();
                        queue.add(prefixXOR[i][j]);
                    }
                }
            }
        }
        return queue.poll();
    }
    public List<String> topKFrequent(String[] words, int k) {
        TreeMap<String,Integer> table = new TreeMap<>();
        for (String s : words){
            table.put(s,table.getOrDefault(s,0)+1);
        }
        PriorityQueue<String> queue = new PriorityQueue<>(k,(o1, o2) -> {
        if (table.get(o1)-table.get(o2)==0){
            return o2.compareTo(o1);
        }else {
            return table.get(o1)-table.get(o2);
        }
        });
        List<String> answer = new LinkedList<>();
        for (Map.Entry<String,Integer> entry : table.entrySet()){
            if(queue.size()<k){
                queue.add(entry.getKey());
            }else{
                String temp = queue.peek();
                if(table.get(temp)<entry.getValue()){
                    queue.poll();
                    queue.add(entry.getKey());
                }
            }
        }
        int s = queue.size();
        for (int i = 0; i < s; i++){
            answer.add(queue.poll());
        }
        Collections.reverse(answer);
        return answer;
    }
    public int longestCommonSubsequence(String text1, String text2) {
        int length1 = text1.length();
        int length2 = text2.length();
        int[][] dp = new int[length1+1][length2+1];
        for (int i = 1; i < length1 + 1; i++){
            for (int j = 1; j < length2 + 1; j++){
                if (text1.charAt(i-1)==text2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[length1][length2];
    }
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int length1 = nums1.length;
        int length2 = nums2.length;
        int[][] dp = new int[length1+1][length2+1];
        for (int i = 1; i < length1 + 1; i++){
            for (int j = 1; j < length2 + 1; j++){
                if (nums1[i-1]==nums2[j-1]){
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i-1][j-1],Math.max(dp[i-1][j],dp[i][j-1]));
                }
            }
        }
        return dp[length1][length2];
    }
    public boolean xorGame(int[] nums) {
        int xorSum = 0;
        for (int i = 0; i < nums.length; i++){
            xorSum = xorSum^nums[i];
        }
        if (xorSum==0) {
            return true;
        }
        return nums.length%2==0;
    }
    public boolean checkZeroOnes(String s) {
        int length1 = 0;
        int length0 = 0;
        int temp1 = 0;
        int temp0 = 0;
        for (int i = 0; i < s.length(); i++){
            if (s.charAt(i)=='1'){
                temp1++;
                length0 = Math.max(length0,temp0);
                temp0 = 0;
            }else {
                temp0++;
                length1 = Math.max(temp1,length1);
                temp1 = 0;
            }
        }
        length0 = Math.max(length0,temp0);
        length1 = Math.max(temp1,length1);
        return length1>length0;
    }
    public int minSpeedOnTime(int[] dist, double hour) {
        if (Math.ceil(hour)<dist.length){
            return -1;
        }

        int left = 1;
        int right = 10000000;
        while (left < right){
            int middle = left + (right - left)/2;
            if (canArrive(dist,hour,middle)){
                right = middle;
            }else {
                left = middle+1;
            }
        }
        return left;
    }
    private boolean canArrive(int[] dist, double hour, int speed){
        double temp = 0;
        for (int i = 0; i < dist.length-1; i++){
            temp += Math.ceil((double) dist[i]/speed);
        }
        temp += (double) dist[dist.length-1]/speed;
        return temp <= hour;
    }
    public boolean canReach(String s, int minJump, int maxJump) {
        int len = s.length();
        if (s.charAt(len-1)=='1'){
            return false;
        }
        char[] a = s.toCharArray();
        int[] dp = new int[len+1];
        int[] prefixSum = new int[len+1];
        Arrays.fill(dp,1);
        dp[1] = 0;
        prefixSum[1] = 0;
        for (int i = 2; i < len + 1; i++){
            if (a[i-1]=='0'){
                if (i-minJump>=0){
                    int left = i - minJump;
                    int right = Math.max(i - maxJump,1);
                    if (prefixSum[left]-prefixSum[right-1]<left-right+1){
                        dp[i] = 0;
                    }
                }
            }
            prefixSum[i] = dp[i] + prefixSum[i-1];
        }
        return dp[len] == 0;
    }

    // static 成员整个类独一份，只有在类首次加载时才会创建，因此只会被 new 一次
    static int N = (int)1e7;
    static int[][] trie = new int[N][2];
    static int idx = 0;
    // 每跑一个数据，会被实例化一次，每次实例化的时候被调用，做清理工作
    public void Solution() {
        for (int i = 0; i <= idx; i++) {
            Arrays.fill(trie[i], 0);
        }
        idx = 0;
    }
    void add(int x) {
        int p = 0;
        for (int i = 31; i >= 0; i--) {
            int u = (x >> i) & 1;
            if (trie[p][u] == 0) trie[p][u] = ++idx;
            p = trie[p][u];
        }
    }
    int getVal(int x) {
        int ans = 0;
        int p = 0;
        for (int i = 31; i >= 0; i--) {
            int a = (x >> i) & 1, b = 1 - a;
            if (trie[p][b] != 0) {
                ans |= (b << i);
                p = trie[p][b];
            } else {
                ans |= (a << i);
                p = trie[p][a];
            }
        }
        return ans;
    }
    public int findMaximumXOR(int[] nums) {
        int ans = 0;
        for (int i : nums) {
            add(i);
            int j = getVal(i);
            ans = Math.max(ans, i ^ j);
        }
        return ans;
    }
    public int[] maximizeXor(int[] nums, int[][] queries) {
        Arrays.sort(nums);
        HashMap<int[], Integer> map = new HashMap<>();
        for (int i = 0; i < queries.length; i++) {
            map.put(queries[i], i);
        }

        Arrays.sort(queries,(o1,o2)->o1[1]-o2[1]);
        int[] answer = new int[queries.length];
        int j = 0;
        for (int i = 0; i < queries.length; i++){
            while (j<nums.length&&nums[j]<=queries[i][1]){
                add(nums[j++]);
            }
            if (j==0){
                answer[map.get(queries[i])] = -1;
            }else {
                answer[map.get(queries[i])] = queries[i][0] ^ getVal(queries[i][0]);
            }
        }
        return answer;
    }
    public int strangePrinter(String s) {
        int length = s.length();
        int[][] dp = new int[length+1][length+1];
        for (int i = 0; i < length+1; i++){
            Arrays.fill(dp[i],Integer.MAX_VALUE);
            dp[i][i] = 1;
        }
        for (int i = length; i >0; i--){
            for (int j = i + 1; j < length + 1; j++){
                if (s.charAt(i-1)==s.charAt(j-1)){
                    dp[i][j] = dp[i][j-1];
                }else {
                    for(int k = i; k < j; k++) {
                        dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k + 1][j]);
                    }
                }
            }
        }
        return dp[1][length];
    }
    public String reverseParentheses(String s) {
        int n = s.length();
        Stack<Integer> stack = new Stack<>();
        int[] pair = new int[n];

        //先去找匹配的括号
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (s.charAt(i) == ')') {
                int j = stack.pop();
                pair[i] = j;
                pair[j] = i;
            }
        }

        StringBuilder res = new StringBuilder();
        // i是当前位置 | d是方向,1就是向右穿
        for (int i = 0, d = 1; i < n; i+=d) {
            if (s.charAt(i) == '(' || s.charAt(i) == ')') {
                // 如果碰到括号，那么去他对应的括号，并且将方向置反
                i = pair[i];
                d = -d;
            } else {
                res.append(s.charAt(i));
            }
        }
        return res.toString();
    }
    public int totalHammingDistance(int[] nums) {
        int[][] table = new int[nums.length][32];
        int answer = 0;
        for (int i = 0; i < nums.length; i++){
            char[] as = Integer.toBinaryString(nums[i]).toCharArray();
            int c = 31;
            for (int j = as.length-1 ; j >= 0; j--,c--){
                table[i][c] = as[j]-48;
            }
        }
        for (int i = 31; i >= 0; i--){
            int temp = table[0][i];
            int count0 = 0;
            int count1 = 0;
            for (int j = 0; j < nums.length; j++){
                if (table[j][i]!=temp){
                    count0++;
                }else {
                    count1++;
                }
            }
            answer += count0*count1;
        }
        return answer;
    }
    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int[][] sum = new int[matrix.length + 1][matrix[0].length + 1];
        int res = 0;
        for (int i = 1; i <= matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                sum[i][j] = sum[i][j - 1] + sum[i - 1][j] - sum[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
        for (int i = 1; i <= matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                for (int p = 1; p <= i; p++) {
                    for (int q = 1; q <= j; q++) {
                        if (sum[i][j] - sum[p - 1][j] - sum[i][q - 1] + sum[p - 1][q - 1] == target) {
                            res++;
                        }
                    }
                }
            }
        }
        return res;
    }
    public boolean isSumEqual(String firstWord, String secondWord, String targetWord) {
        char[] first = firstWord.toCharArray();
        char[] second = secondWord.toCharArray();
        char[] tgt = targetWord.toCharArray();
        StringBuilder firstw = new StringBuilder();
        StringBuilder secondw = new StringBuilder();
        StringBuilder tgtw = new StringBuilder();
        for (char value : first) {
            firstw.append(value - 97);
        }
        for (char value : second) {
            secondw.append(value - 97);
        }
        for (char value : tgt) {
            tgtw.append(value - 97);
        }
        return Integer.parseInt(firstw.toString())+Integer.parseInt(secondw.toString())==Integer.parseInt(tgtw.toString());
    }
    public String maxValue(String n, int x) {
        char[] chars = n.toCharArray();
        LinkedList<Character> answer = new LinkedList<>();
        char s = (char) (x+48);
        int c = 0;
        if (chars[0]!='-'){
            for (int i = 0; i < chars.length; i++){
                if ((chars[i]-48<x&&c==0)){
                    answer.add(s);
                    c=1;
                }
                answer.add(chars[i]);
            }
            if (c==0){
                answer.add(s);
            }
        }else {
            answer.add('-');
            for (int i = 1; i < chars.length; i++){
                if ((chars[i]-48>x&&c==0)){
                    answer.add(s);
                    c=1;
                }
                answer.add(chars[i]);
            }
            if (c==0){
                answer.add(s);
            }
        }
        StringBuilder res = new StringBuilder();
        for (char p : answer){
            res.append(p);
        }
        return res.toString();
    }
    public int[] assignTasks(int[] servers, int[] tasks) {
        PriorityQueue<int[]> available = new PriorityQueue<>((o1, o2) -> {
            if (o1[0]==o2[0]){
                return o1[1]-o2[1];
            }else {
                return o1[0]-o2[0];
            }
        });
        PriorityQueue<int[]> unavailable = new PriorityQueue<>((o1, o2) -> {
            if (o1[0]==o2[0]){
                return o1[1]-o2[1];
            }else {
                return o1[0]-o2[0];
            }
        });
        int timestamp = 0;
        int[] answer = new int[tasks.length];
        for (int i =0; i < servers.length; i++){
            available.add(new int[]{servers[i],i});
        }

        for (int i = 0; i < tasks.length; i++){
            timestamp = Math.max(timestamp,i);
            while (!unavailable.isEmpty()&&unavailable.peek()[0]<=timestamp){
                int temp = unavailable.poll()[1];
                available.add(new int[]{servers[temp],temp});
            }
            if (available.isEmpty()){
                timestamp = unavailable.peek()[0];
                while (!unavailable.isEmpty()&&unavailable.peek()[0]<=timestamp){
                    int temp = unavailable.poll()[1];
                    available.add(new int[]{servers[temp],temp});
                }
            }
            int index = available.poll()[1];
            unavailable.add(new int[]{tasks[i]+timestamp,index});
            answer[i] = index;
        }
        return answer;
    }
    public boolean isPowerOfTwo(int n) {
        if(n<=0){
            return false;
        }
        int count = 0;
        for (int i = 0; i<32;i++){
            if (((n>>i)&1)==1){
                count++;
            }
        }
        return count==1;
    }
    public boolean isPowerOfFour(int n) {
        if (n<=0){
            return false;
        }
        int count = 0;
        int c = 0;
        for (int i = 0; i<32;i++){
            if (((n>>i)&1)==1){
                count++;
                if (i%2==0){
                    c++;
                }
            }
        }
        return c==1&&c==count;
    }
    public boolean checkSubarraySum(int[] nums, int k) {
        int[] prefix = new int[nums.length+1];
        HashMap<Integer,Integer> map = new HashMap<>();
        map.put(0,-1);
        for (int i = 1; i < nums.length+1; i++){
            prefix[i] = prefix[i-1] + nums[i-1];
            int temp = prefix[i]%k;
            if (map.containsKey(temp)){
                if ((i-1)-map.get(temp)>=2){
                    return true;
                }
            }else {
                map.put(temp,i-1);
            }
        }
        return false;
    }
    public int findMaxLength(int[] nums) {
        int[] prefix = new int[nums.length+1];
        HashMap<Integer,Integer> map = new HashMap<>();
        int answer = 0;
        map.put(0,-1);
        for (int i = 1; i < nums.length+1; i++){
            if (nums[i-1]==0){
                prefix[i] = prefix[i-1] -1;
            }else {
                prefix[i] = prefix[i-1] +1;
            }

            if (map.containsKey(prefix[i])){
                if ((i-1)-map.get(prefix[i])>=2){
                    answer = Math.max(answer,(i-1)-map.get(prefix[i]));
                }
            }else {
                map.put(prefix[i], i-1);
            }
        }
        return answer;
    }
    public boolean findRotation(int[][] mat, int[][] target) {
        int l = mat.length;
        for (int o = 0; o < 4; o++) {
            int p = 0;
            for (int i = 0; i < l; i++){
                for (int j = 0; j < l; j++){
                    if (mat[i][j]!=target[i][j]){
                        p=1;
                        break;
                    }
                }
            }
            if (p==0){
                return true;
            }
            for (int i = 0; i < l / 2; i++) {
                for (int j = 0; j < l; j++) {
                    int temp = mat[i][j];
                    mat[i][j] = mat[l - i - 1][j];
                    mat[l - i - 1][j] = temp;
                }
            }
            for (int i = 0; i < l; i++) {
                for (int j = 0; j < i; j++) {
                    int temp = mat[i][j];
                    mat[i][j] = mat[j][i];
                    mat[j][i] = temp;
                }
            }
        }
        return false;
    }
    public int reductionOperations(int[] nums) {
        Map<Integer,Integer> map = new TreeMap<>(((o1, o2) -> o2-o1));
        for (int n : nums){
            map.put(n,map.getOrDefault(n,0)+1);
        }
        int answer = 0;
        int min = Integer.MAX_VALUE;
        for (Map.Entry<Integer,Integer> entry : map.entrySet()){
            min = Math.min(min,entry.getKey());
        }
        int count = 0;
        for (Map.Entry<Integer,Integer> entry : map.entrySet()){
            if (entry.getKey()==min){
                break;
            }
            answer = answer+entry.getValue()+count;
            count+=entry.getValue();
        }
        return answer;
    }
    public int minFlips(String ss) {
        int len = ss.length();
        char[] s = ss.toCharArray();
        int[][] dp = new int[2][len + 1];
        dp[0][0] = 0;
        dp[1][0] = 0;
        for (int i = 1; i < len + 1; i++) {

            if (s[i - 1] == '0'){
                dp[0][i] = dp[1][i - 1];
                dp[1][i] = dp[0][i - 1] + 1;
            }else if (s[i - 1] == '1'){
                dp[1][i] = dp[0][i - 1];
                dp[0][i] = dp[1][i - 1] + 1;
            }

        }
        return Math.min(dp[0][len],dp[1][len]);
    }
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        int[][][] dp = new int[len + 1][m + 1][n + 1];

        for (int i = 1; i <= len; i++) {
            // 注意：有一位偏移
            int[] count = countZeroAndOne(strs[i - 1]);
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n; k++) {
                    // 先把上一行抄下来
                    dp[i][j][k] = dp[i - 1][j][k];
                    int zeros = count[0];
                    int ones = count[1];
                    if (j >= zeros && k >= ones) {
                        dp[i][j][k] = Math.max(dp[i - 1][j][k], dp[i - 1][j - zeros][k - ones] + 1);
                    }
                }
            }
        }
        return dp[len][m][n];
    }
    private int[] countZeroAndOne(String str) {
        int[] cnt = new int[2];
        for (char c : str.toCharArray()) {
            cnt[c - '0']++;
        }
        return cnt;
    }

    public int lastStoneWeightII(int[] stones) {
        int n = stones.length;
        int sum = 0;
        for (int i : stones) {
            sum += i;
        }
        int t = sum / 2;
        int[][] f = new int[n + 1][t + 1];
        for (int i = 1; i <= n; i++) {
            int x = stones[i - 1];
            for (int j = 0; j <= t; j++) {
                f[i][j] = f[i - 1][j];
                if (j >= x) {
                    f[i][j] = Math.max(f[i][j], f[i - 1][j - x] + x);
                }
            }
        }
        return Math.abs(sum - f[n][t] - f[n][t]);
    }
    public int change1(int cnt, int[] cs) {
        int n = cs.length;
        int[][] f = new int[n + 1][cnt + 1];
        f[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            int val = cs[i - 1];
            for (int j = 0; j <= cnt; j++) {
                f[i][j] = f[i - 1][j];
                for (int k = 1; k * val <= j; k++) {
                    f[i][j] += f[i - 1][j - k * val];
                }
            }
        }
        return f[n][cnt];
    }
    public int change2(int amount, int[] coins) {
        int len = coins.length;
        if (len == 0) {
            if (amount == 0) {
                return 1;
            }
            return 0;
        }

        int[][] dp = new int[len][amount + 1];
        // 题解中有说明应该如何理解这个初始化
        dp[0][0] = 1;

        // 填第 1 行
        for (int i = coins[0]; i <= amount; i += coins[0]) {
            dp[0][i] = 1;
        }

        for (int i = 1; i < len; i++) {
            for (int j = 0; j <= amount; j++) {
                for (int k = 0; j - k * coins[i] >= 0; k++) {
                    dp[i][j] += dp[i - 1][j - k * coins[i]];
                }
            }
        }
        return dp[len - 1][amount];
    }
    public String largestNumber(int[] cost, int t) {
        int[] f = new int[t + 1];
        Arrays.fill(f, Integer.MIN_VALUE);
        f[0] = 0;
        for (int i = 1; i <= 9; i++) {
            int u = cost[i - 1];
            for (int j = u; j <= t; j++) {
                f[j] = Math.max(f[j], f[j - u] + 1);
            }
        }
        if (f[t] < 0) return "0";
        String ans = "";
        for (int i = 9, j = t; i >= 1; i--) {
            int u = cost[i - 1];
            while (j >= u && f[j] == f[j - u] + 1) {
                ans += String.valueOf(i);
                j -= u;
            }
        }
        return ans;
    }
    public boolean makeEqual(String[] words) {
        HashMap<Character,Integer> map = new HashMap<>();
        if (words.length==1){
            return true;
        }
        for (String s : words){
            for (int i = 0; i < s.length(); i++){
                map.put(s.charAt(i),map.getOrDefault(s.charAt(i),0)+1);
            }
        }
        for (Map.Entry<Character,Integer> entry : map.entrySet()){
            if (entry.getValue()%words.length!=0){
                return false;
            }
        }
        return true;
    }
    public int maximumRemovals(String s, String p, int[] removable) {
        int low = 0;
        int high = removable.length+1;

        while (low<high){
            int middle = low+(high-low)/2;
            if (isSer(middle,removable,s,p)){

                low = middle+1;
            }else {
                high = middle;
            }
        }
        return low-1;
    }
    private boolean isSer(int middle,int[] remove,String s,String p){
        int[] newRemove = new int[middle];
        System.arraycopy(remove,0,newRemove,0,middle);
        Arrays.sort(newRemove);
        int k = 0;
        int j = 0;
        for (int i = 0; i < s.length(); i++){
            if (j==p.length()){
                return true;
            }
            if (k==middle||i!=newRemove[k]){
                if (p.charAt(j)==s.charAt(i)){
                    j++;
                }
            }else if (i==newRemove[k]){
                k++;
            }
        }
        return j==p.length();
    }
    public boolean mergeTriplets(int[][] triplets, int[] target) {
        int a = 0;
        int b = 0;
        int c = 0;

        for (int[] triplet : triplets) {
            if (triplet[0]<=target[0]&&triplet[1]<=target[1]&&triplet[2]<=target[2]){
                a = Math.max(a,triplet[0]);
                b = Math.max(b,triplet[1]);
                c = Math.max(c,triplet[2]);
            }
        }
        return a==target[0] && b==target[1] && c==target[2];
    }
    public int peakIndexInMountainArray(int[] arr) {
        int left = 1;
        int right = arr.length-2;
        while (left<right){
            int middle = left + (right-left)/2;
            if (arr[middle]>arr[middle-1]&&arr[middle]>arr[middle+1]){
                return middle;
            }else if (arr[middle]<arr[middle-1]&&arr[middle]>arr[middle+1]){
                right=middle;
            }else if (arr[middle]>arr[middle-1]&&arr[middle]<arr[middle+1]){
                left=middle+1;
            }
        }
        return left;
    }

    public int make(char c) {
        switch(c) {
            case ' ': return 0;
            case '+':
            case '-': return 1;
            case '.': return 3;
            case 'e':
            case 'E': return 4;
            default:
                if(c >= 48 && c <= 57) return 2;
        }
        return -1;
    }
    public boolean isNumber(String s) {
        int state = 0;
        int finals = 0b101101000;
        int[][] transfer = new int[][]{
                { 0, 1, 6, 2,-1},
                {-1,-1, 6, 2,-1},
                {-1,-1, 3,-1,-1},
                { 8,-1, 3,-1, 4},
                {-1, 7, 5,-1,-1},
                { 8,-1, 5,-1,-1},
                { 8,-1, 6, 3, 4},
                {-1,-1, 5,-1,-1},
                { 8,-1,-1,-1,-1}
        };
        char[] ss = s.toCharArray();
        for(int i=0; i < ss.length; ++i) {
            int id = make(ss[i]);
            if (id < 0) {
                return false;
            }
            state = transfer[state][id];
            if (state < 0) {
                return false;
            }
        }
        return (finals & (1 << state)) > 0;
    }
    public String smallestGoodBase(String n) {
        long tgt = Long.parseLong(n);

        double value = Math.log(tgt) / Math.log(2);

        for (int i = (int) Math.ceil(value); i > 0; i--) {
            long left = 2;
            long right = (long) Math.ceil(Math.pow(tgt, 1.0 / (i - 1)));
            if (i-1==1){
                 right = tgt;
            }
            while (left < right) {
                long middle = (left + right) / 2;
                long sum = 1;
                for (int j = 0; j < i-1; j++) {
                    sum = sum * middle + 1;
                }
                if (sum==tgt){
                    return String.valueOf(middle);
                }else if (sum>tgt){
                    right = middle;
                }else if (sum<tgt){
                    left = middle+1;
                }
            }
        }
        return "";
    }
    static Map<Integer, Integer> map = new HashMap<>();
    int get(int cur) {
        if (map.containsKey(cur)) {
            return map.get(cur);
        }
        int ans = 0;
        for (int i = cur; i > 0; i -= lowbit(i)) ans++;
        map.put(cur, ans);
        return ans;
    }
    int lowbit(int x) {
        return x & -x;
    }

    int nl;
    int ansl = Integer.MIN_VALUE;
    int[] hash;

    public int maxLength(List<String> arr) {
        nl = arr.size();
        HashSet<Integer> set = new HashSet<>();
        for (String s : arr) {
            int val = 0;
            for (char c : s.toCharArray()) {
                int t = (int)(c - 'a');
                if (((val >> t) & 1) != 0) {
                    val = -1;
                    break;
                }
                val |= (1 << t);
            }
            if (val != -1) set.add(val);
        }

        nl = set.size();
        if (nl == 0) return 0;
        hash = new int[nl];

        int idx = 0;
        int total = 0;
        for (Integer i : set) {
            hash[idx++] = i;
            total |= i;
        }
        dfs(0, 0, total);
        return ansl;
    }
    void dfs(int u, int cur, int total) {
        if (get(cur | total) <= ansl) return;
        if (u == nl) {
            ansl = Math.max(ansl, get(cur));
            return;
        }
        // 在原有基础上，选择该数字（如果可以）
        if ((hash[u] & cur) == 0) {
            dfs(u + 1, hash[u] | cur, total - (total & hash[u]));
        }
        // 不选择该数字
        dfs(u + 1, cur, total);
    }
    public String largestOddNumber(String num) {
        String answer = "";
        for (int i = num.length()-1; i >=0; i--){
            if ((num.charAt(i)-'0')%2!=0){
                answer = num.substring(0,i+1);
                break;
            }
        }
        return answer;
    }
    public int numberOfRounds(String startTime, String finishTime) {
        int start = Integer.parseInt(startTime.substring(0,2));
        int end = Integer.parseInt(finishTime.substring(0,2));
        int answer = 0;
        if (start==end){
            int a = Integer.parseInt(startTime.substring(3,5));
            int b = Integer.parseInt(finishTime.substring(3,5));
            if (a<b) {
                int j = b - a;
                if (j < 15) {
                    return 0;
                }
                answer = j / 15;
                for (int i = 0; i < 60 && j < 60; i++) {
                    if (i == a) {
                        break;
                    }
                    if (i % 15 == 0) {
                        answer--;
                    }
                    if (j % 15 == 0) {
                        answer++;
                    }
                    j++;
                }
            }else {
                if (0<a&&a<=15){
                    a=1;
                }else if (15<a&&a<=30){
                    a=2;
                }else if (30<a&&a<=45){
                    a=3;
                }else if (45<a&&a<60){
                    a=4;
                }
                answer = 96-a+b/15;
            }
        }else if (start > end){
            int a = Integer.parseInt(startTime.substring(3,5));
            int b = Integer.parseInt(finishTime.substring(3,5));
            answer = (24-start)*4-(4-(60-a)/15)+end*4+b/15;
        }else if (start < end){
            int a = Integer.parseInt(startTime.substring(3,5));
            int b = Integer.parseInt(finishTime.substring(3,5));
            answer = (end-start)*4-(4-(60-a)/15)+b/15;
        }
        return answer;
    }
    private static final int[][] DIRECTIONS = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
    private boolean[][] visited;
    private boolean[][] g1;
    private boolean singal = false;
    private int rows;
    private int cols;
    private int[][] grid2;
    public int countSubIslands(int[][] grid1, int[][] grid2) {
        rows = grid2.length;
        if (rows == 0) {
            return 0;
        }
        cols = grid2[0].length;

        this.grid2 = grid2;
        visited = new boolean[rows][cols];
        g1 = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                g1[i][j] = grid1[i][j] == 1;
            }
        }
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // 如果是岛屿中的一个点，并且没有被访问过，就进行深度优先遍历
                if (!visited[i][j] && grid2[i][j] == 1) {
                    bfs(i, j);
                    count++;
                    if (singal){
                        count--;
                        singal = false;
                    }
                }
            }
        }
        return count;
    }
    private void bfs(int i, int j) {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(i * cols + j);
        // 注意：这里要标记上已经访问过
        visited[i][j] = true;
        if (!g1[i][j]){
            singal = true;
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            int curX = cur / cols;
            int curY = cur % cols;
            for (int k = 0; k < 4; k++) {
                int newX = curX + DIRECTIONS[k][0];
                int newY = curY + DIRECTIONS[k][1];
                if (inArea(newX, newY) && grid2[newX][newY] == 1 && !visited[newX][newY]) {
                    queue.offer(newX * cols + newY);
                    // 特别注意：在放入队列以后，要马上标记成已经访问过，语义也是十分清楚的：反正只要进入了队列，迟早都会遍历到它
                    // 而不是在出队列的时候再标记，如果是出队列的时候再标记，会造成很多重复的结点进入队列，造成重复的操作，这句话如果你没有写对地方，代码会严重超时的
                    visited[newX][newY] = true;
                    if (!g1[newX][newY]){
                        singal = true;
                    }
                }
            }
        }
    }
    public List<String> readBinaryWatch(int turnedOn) {
        List<String> answer = new ArrayList<>();
        if (turnedOn>8){
            return answer;
        }
        for (int i = 0; i < 12; i++){
            for (int j = 0; j < 60; j++){
                if (Integer.bitCount(i)+Integer.bitCount(j)==turnedOn){
                    if (j<10){
                        answer.add(i + ":0" + j);
                    }else {
                        answer.add(i + ":" + j);
                    }
                }
            }
        }
        return answer;
    }
    public int maxPoints(int[][] points) {
        int answer = 1;
        for (int i = 0; i < points.length; i++){
            for (int j = 0; j < points.length; j++){
                if (i!=j){
                    int temp = 2;
                    for (int k = 0; k < points.length; k++){
                        if (k!=i&&k!=j){
                            if ((points[k][0]-points[i][0])*(points[j][1]-points[i][1])==(points[j][0]-points[i][0])*(points[k][1]-points[i][1])){
                                temp++;
                            }
                        }
                    }
                    answer = Math.max(answer,temp);
                }
            }
        }
        return answer;
    }
    public int openLock(String[] deadends, String target) {
        Set<String> set = new HashSet<>(Arrays.asList(deadends));
        //开始遍历的字符串是"0000"，相当于根节点
        String startStr = "0000";
        if (set.contains(startStr))
            return -1;
        //创建队列
        Queue<String> queue = new LinkedList<>();
        //记录访问过的节点
        Set<String> visited = new HashSet<>();
        queue.offer(startStr);
        visited.add(startStr);
        //树的层数
        int level = 0;
        while (!queue.isEmpty()) {
            //每层的子节点个数
            int size = queue.size();
            while (size-- > 0) {
                //每个节点的值
                String str = queue.poll();
                //对于每个节点中的4个数字分别进行加1和减1，相当于创建8个子节点，这八个子节点
                if (str.equals(target))
                    return level;
                //可以类比二叉树的左右子节点
                for (int i = 0; i < 4; i++) {
                    char ch = str.charAt(i);
                    //strAdd表示加1的结果，strSub表示减1的结果
                    String strAdd = str.substring(0, i) + (ch == '9' ? 0 : ch - '0' + 1) + str.substring(i + 1);
                    String strSub = str.substring(0, i) + (ch == '0' ? 9 : ch - '0' - 1) + str.substring(i + 1);
                    //如果找到直接返回

                    //不能包含死亡数字也不能包含访问过的字符串
                    if (!visited.contains(strAdd) && !set.contains(strAdd)) {
                        queue.offer(strAdd);
                        visited.add(strAdd);
                    }
                    if (!visited.contains(strSub) && !set.contains(strSub)) {
                        queue.offer(strSub);
                        visited.add(strSub);
                    }
                }
            }
            //当前层访问完了，到下一层，层数要加1
            level++;
        }
        return -1;
    }
    public int slidingPuzzle(int[][] board) {
        int[][] answer = new int[][]{{1,2,3},{4,5,0}};
        int[][] DIRECTIONS = new int[][]{{1,0},{0,1},{-1,0},{0,-1}};

        Queue<int[][]> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        queue.offer(board);
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < 2; i++){
            for (int j = 0; j < 3; j++){
                b.append(board[i][j]);
            }
        }
        visited.add(b.toString());

        int level = 0;

        while (!queue.isEmpty()) {

            int size = queue.size();
            while (size-- > 0) {
                int[][] temp;
                temp = queue.poll();
                int x = 0;
                int y = 0;
                for (int i = 0; i < 2; i++){
                    for (int j = 0; j < 3; j++){
                        if (temp[i][j]==0){
                            y = i;
                            x = j;
                            break;
                        }
                    }
                }
                if (Arrays.deepEquals(temp, answer)) {
                    return level;
                }
                for (int i = 0; i < 4; i++){
                    int[][] newTemp = new int[2][3];
                    int tx = x+DIRECTIONS[i][0];
                    int ty = y+DIRECTIONS[i][1];
                    if ((tx<3&&tx>=0)&&(ty<2&&ty>=0)){
                        int t = temp[ty][tx];

                        StringBuilder a = new StringBuilder();
                        for (int k = 0; k < 2; k++){
                            System.arraycopy(temp[k], 0, newTemp[k], 0, 3);
                        }
                        newTemp[ty][tx] = 0;
                        newTemp[y][x] = t;
                        for (int k = 0; k < 2; k++){
                            for (int l = 0; l < 3; l++){
                                a.append(newTemp[k][l]);
                            }
                        }

                        if (!visited.contains(a.toString())){
                            queue.offer(newTemp);
                            visited.add(a.toString());
                        }
                    }
                }
            }
            level++;
        }
        return -1;
    }
    public int snakesAndLadders(int[][] board) {
        int n = board.length;
        int[] nums = new int[n * n + 1];
        boolean isRight = true;
        for (int i = n - 1, idx = 1; i >= 0; i--) {
            for (int j = (isRight ? 0 : n - 1); isRight ? j < n : j >= 0; j += isRight ? 1 : -1) {
                nums[idx++] = board[i][j];
            }
            isRight = !isRight;
        }
        Deque<Integer> d = new ArrayDeque<>();
        Map<Integer, Integer> m = new HashMap<>();
        d.addLast(1);
        m.put(1, 0);
        while (!d.isEmpty()) {
            int poll = d.pollFirst();
            int step = m.get(poll);
            if (poll == n * n)
                return step;
            for (int i = 1; i <= 6; i++) {
                int np = poll + i;
                if (np > n * n)
                    continue;
                if (nums[np] != -1)
                    np = nums[np];
                if (m.containsKey(np))
                    continue;
                m.put(np, step + 1);
                d.addLast(np);
            }
        }
        return -1;
    }
    public int maxProductDifference(int[] nums) {
        Arrays.sort(nums);
        return (nums[nums.length-1]*nums[nums.length-2])-(nums[0]*nums[1]);
    }
    public int[][] rotateGrid(int[][] grid, int k) {
        int level = 0;
        int left = -1;
        int right = grid[0].length+1;
        int top = -1;
        int bottom = grid.length+1;

        while (level<Math.min(grid[0].length,grid.length)/2){
            int tempk;
            left = ++left ;
            right = --right ;
            top = ++top;
            bottom = --bottom;
            int length = (right-left-1+bottom-top-1) * 2;
            level++;
            tempk = k % length;
            for (int i = 0; i < tempk; i++){

                int leftUpConner = grid[top][left];
                for (int a = left; a < right-1; a++){
                    grid[top][a] = grid[top][a+1];
                }

                int leftDownConner = grid[bottom-1][left];
                for (int a = bottom-1; a > top+1; a--){
                    grid[a][left] = grid[a-1][left];
                }
                grid[top+1][left] = leftUpConner;

                int rightDownConner = grid[bottom-1][right-1];
                for (int a = right-1; a > left+1; a--){
                    grid[bottom-1][a] = grid[bottom-1][a-1];
                }
                grid[bottom-1][left+1] = leftDownConner;

                for (int a = top; a < bottom-1; a++){
                    grid[a][right-1] = grid[a+1][right-1];
                }
                grid[bottom-2][right-1] = rightDownConner;
            }
        }
        return grid;
    }
    
}
