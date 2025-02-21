package com.z992907759.zhao.controller;

import com.z992907759.zhao.base.ResponseResult;
import com.z992907759.zhao.service.LoginService;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import io.swagger.annotations.Api;

@Slf4j
@Api(tags = "登陆")
@RestController
@RequestMapping
public class LogginController {

    @Autowired
    LoginService loginService;

    @ApiOperation("分页")
    @GetMapping("/login")
    public ResponseResult LoginController() {
        loginService.login();
        return ResponseResult.success("登录成功");
    }

}
