package com.z992907759.zhao.service.impl;

import com.z992907759.zhao.mapper.LoginMapper;
import com.z992907759.zhao.service.LoginService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class LoginServiceImpl implements LoginService {

    @Autowired
    LoginMapper loginMapper;

    @Override
    public void login() {

    }
}
