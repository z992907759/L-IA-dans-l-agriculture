package com.z992907759.zhao.enums;


import com.z992907759.zhao.base.IBasicEnum;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 基础枚举
 *
 * @author itcast
 */
@Getter
@AllArgsConstructor
public enum BasicEnum implements IBasicEnum {

    SUCCEED(200, "操作成功"),
    SECURITY_ACCESSDENIED_FAIL(401, "权限不足!"),
    LOGIN_FAIL(401, "用户登录失败"),
    LOGIN_LOSE_EFFICACY(401, "登录状态失效，请重新登录"),
    SYSYTEM_FAIL(500, "系统运行异常"),


    //权限相关异常：1400-1499
    DEPT_DEPTH_UPPER_LIMIT(1400, "部门最多4级"),
    PARENT_DEPT_DISABLE(1401, "父级部门为禁用状态,不允许启用"),
    DEPT_NULL_EXCEPTION(1402, "部门不能为空"),
    POSITION_DISTRIBUTED(1403, "职位已分配，不允许禁用"),
    MENU_NAME_DUPLICATE_EXCEPTION(1404, "菜单路由重复"),
    MENU_PATH_DUPLICATE_EXCEPTION(1405, "菜单路由重复"),
    RESOURCE_DEPTH_UPPER_LIMIT(1406, "菜单最多3级"),
    USER_ROLE_AND_MENU_EMPTY(1407, "请为用户分配角色和菜单"),
    PARENT_MENU_DISABLE(1408, "父级菜单为禁用状态,不允许启用"),
    MENU_DISTRIBUTED(1409, "菜单已分配，不允许禁用"),
    BUTTON_DISTRIBUTED(1410, "按钮已分配，不允许禁用"),
    ROLE_DISTRIBUTED(1411, "该角色已分配用户,不能删除"),
    USER_LOCATED_BOTTOMED_DEPT(1412, "请选择最底层部门"),
    DEPT_BINDING_USER(1413, "部门已绑定用户,不允许禁用"),
    USER_EMPTY_EXCEPTION(1414, "用户不存在，不能启用或禁用"),
    ORIGINAL_PASSWORD_ERROR(1415, "原密码不正确，请重新输入"),
    ORIGINAL_CANNOT_EQUAL_NEW(1416, "新密码和旧密码不能一致"),
    GET_OPENID_ERROR(1417, "小程序登录，获取openId失败"),
    GET_PHONE_ERROR(1418, "小程序登录，获取手机号失败"),
    GET_TOKEN_ERROR(1419, "小程序登录，获取token失败"),


    //业务相关异常：1500-1599
    WEBSOCKET_PUSH_MSG_ERROR(1500, "websocket推送消息失败"),
    CLOSE_BALANCE_ERROR(1501, "关闭余额账户失败"),
    MONTH_BILL_DUPLICATE_EXCEPTION(1502, "该老人的月度账单已生成，不可重复生成"),
    MONTH_OUT_CHECKIN_TERM(1503, "该月不在费用期限内"),
    RETREAT_TERM_LACK_MONTH_BILL(1504, "退住日期内缺少月度账单，请前往财务管理模块手动生成"),
    ELDER_ALREADY_CHECKIN(1505, "该老人已入住，请重新输入"),
    CHECKIN_TERM_SHOULD_CONTAIN_COST_TERM(1506, "费用期限应该在入住期限内"),
    DEVICE_NAME_EXIST(1507, "设备名称已存在，请重新输入"),
    LOCATION_BINDING_PRODUCT(1508, "该老人/位置已绑定该产品，请重新选择"),
    IOT_REGISTER_DEVICE_ERROR(1509, "物联网接口 - 注册设备，调用失败"),
    IOT_QUERY_PRODUCT_ERROR(1510, "物联网接口 - 查询产品，调用失败"),
    IOT_QUERY_DEVICE_ERROR(1511, "物联网接口 - 查询产品，调用失败"),
    IOT_QUERY_DEVICE_PROPERTY_STATUS_ERROR(1512, "物联网接口 - 查询设备的物模型运行状态，调用失败"),
    IOT_QUERY_THING_MODEL_PUBLISHED_ERROR(1513, "物联网接口 - 查询物模型数据，调用失败"),
    DEVICE_NOT_EXIST(1514, "该设备不存在，无法修改"),
    IOT_BATCH_UPDATE_DEVICE_ERROR(1515, "物联网接口 - 批量修改设备名称，调用失败"),
    RE_SELECT_PRODUCT(1516, "该老人/位置已绑定该产品，请重新选择"),
    IOT_DELETE_DEVICE_ERROR(1517, "物联网接口 - 删除设备，调用失败"),
    IOT_LIST_PRODUCT_ERROR(1518, "物联网接口 - 查看所有产品列表，调用失败"),
    IOT_OPEN_DOOR_ERROR(1519, "开门失败"),
    IOT_QUERY_DEVICE_SERVICE_DATA_ERROR(1520, "物联网接口 - 调用查询指定设备的服务调用记录接口，调用失败"),
    ELDER_NOT_EXIST(1521, "老人不存在"),
    MEMBER_ALREADY_BINDING_ELDER(1522, "已绑定过此家人"),
    CANNOT_PLACE_ORDER_DUE_ELDER_ALREADY_RETREATED(1523, "已退住，不可下单"),
    ORDER_CLOSED(1524, "订单已关闭"),
    CANNOT_RESERVATION_DUE_ELDER_ALREADY_RETREATED(1525, "已退住，不可预约"),
    RESERVATION_CANCEL_COUNT_UPPER_LIMIT(1526, "今天取消次数已达上限，不可进行预约"),
    TIME_ALREADY_RESERVATED_BY_PHONE(1527, "此手机号已预约该时间"),
    RETREAT_SHOULD_IN_COST_TERM(1528, "请在费用期限内发起退住申请"),
    UPLOAD_FILE_EMPTY(1529, "上传图片不能为空"),
    DONE_ORDER_CANNOT_REFUND(1530, "已执行的订单不可退款"),
    BED_INSERT_FAIL(1531,"床位新增失败"),
    ENABLED_CANNOT_DELETED(1532,"启用状态不能删除"),
    DEVICE_NOTFOUND(1533,"未找到该设备"),

    //支付相关异常：1600-1699
    APPLY_TRADE_FAIL(1600, "发起支付失败"),
    GET_PAYMENT_SIGNATURE_FAIL(1601, "获取支付签名失败"),
    CLOSE_TRADING_FAIL(1602, "关闭交易单失败"),
    UNKNOWN_TRADING_STATUS(1603, "查询交易单获得未知状态"),
    QUERY_TRADING_FAIL(1604, "查询交易单失败"),
    APPLY_REFUND_FAIL(1605, "申请退款失败"),
    QUERY_REFUND_FAIL(1606, "查询退款失败");

    /**
     * 编码
     */
    public final int code;
    /**
     * 信息
     */
    public final String msg;
}

